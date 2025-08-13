// server.js - WebRTC signaling server with CrowdCount API authentication
require('dotenv').config(); // Load .env file
const express = require('express');
const WebSocket = require('ws');
const http = require('http');
const crypto = require('crypto');
const axios = require('axios');

const app = express();
const server = http.createServer(app);
const wss = new WebSocket.Server({ server });

// Store active connections and feeds
const connections = new Map();  // clientId -> {ws, type, feedId, deviceId, password, authenticated}
const feeds = new Map();        // feedId -> {senderId, name, deviceId, viewers: Set}

// CrowdCount API configuration
const CROWDCOUNT_API_URL = process.env.CROWDCOUNT_API_URL || 'https://crowdcount.tech/api';
const USE_CROWDCOUNT_AUTH = process.env.USE_CROWDCOUNT_AUTH !== 'false'; // Default true
const FALLBACK_TO_LOCAL = process.env.FALLBACK_TO_LOCAL === 'true'; // Default false

// Local authentication store (fallback or additional devices)
const localAuthorizedDevices = new Map();

// Environment configuration
const REQUIRE_AUTH = process.env.REQUIRE_AUTH !== 'false'; // Default true
const AUTH_TIMEOUT_MS = parseInt(process.env.AUTH_TIMEOUT_MS) || 10000; // 10 seconds default
const PORT = process.env.PORT || 3000;
const ALLOW_DEVICE_REGISTRATION = process.env.ALLOW_DEVICE_REGISTRATION === 'true'; // Default false

// Load local devices from environment variables (as fallback)
function loadLocalDevicesFromEnv() {
    let deviceCount = 0;
    
    // Try to load up to 20 devices from environment
    for (let i = 1; i <= 20; i++) {
        const deviceId = process.env[`DEVICE_${i}_ID`];
        const devicePassword = process.env[`DEVICE_${i}_PASSWORD`];
        const deviceName = process.env[`DEVICE_${i}_NAME`] || `Device ${i}`;
        
        if (deviceId && devicePassword) {
            localAuthorizedDevices.set(deviceId, {
                password: hashPassword(devicePassword),
                name: deviceName
            });
            deviceCount++;
            console.log(`Loaded local device ${i}: ${deviceId} (${deviceName})`);
        }
    }
    
    // Also support single device for backward compatibility
    if (process.env.DEVICE_ID && process.env.DEVICE_PASSWORD) {
        const deviceId = process.env.DEVICE_ID;
        const devicePassword = process.env.DEVICE_PASSWORD;
        const deviceName = process.env.DEVICE_NAME || 'Default Device';
        
        if (!localAuthorizedDevices.has(deviceId)) {
            localAuthorizedDevices.set(deviceId, {
                password: hashPassword(devicePassword),
                name: deviceName
            });
            deviceCount++;
            console.log(`Loaded default local device: ${deviceId} (${deviceName})`);
        }
    }
    
    // Load admin credentials if provided
    if (process.env.ADMIN_USERNAME && process.env.ADMIN_PASSWORD) {
        app.locals.adminUsername = process.env.ADMIN_USERNAME;
        app.locals.adminPassword = hashPassword(process.env.ADMIN_PASSWORD);
        console.log('Admin credentials loaded');
    }
    
    return deviceCount;
}

// Load local devices on startup
const localDeviceCount = loadLocalDevicesFromEnv();

// Helper function to hash passwords
function hashPassword(password) {
    return crypto.createHash('sha256').update(password).digest('hex');
}

// Verify device against CrowdCount API
async function verifyCrowdCountDevice(deviceId, password) {
    try {
        const response = await axios.get(`${CROWDCOUNT_API_URL}/check_password`, {
            params: {
                user_id: deviceId,
                password: password
            },
            timeout: 5000 // 5 second timeout
        });
        
        if (response.data.status === 'does_not_exist') {
            return { success: false, error: 'Device does not exist in CrowdCount' };
        }
        
        if (response.data.is_password_correct === true) {
            console.log(`Device ${deviceId} authenticated via CrowdCount API`);
            return { success: true, deviceName: deviceId, source: 'crowdcount' };
        } else {
            return { success: false, error: 'Invalid password' };
        }
    } catch (error) {
        console.error(`CrowdCount API error for device ${deviceId}:`, error.message);
        
        // If API is down and fallback is enabled, try local auth
        if (FALLBACK_TO_LOCAL) {
            console.log(`Falling back to local authentication for ${deviceId}`);
            return verifyLocalDevice(deviceId, password);
        }
        
        return { success: false, error: 'CrowdCount API unavailable' };
    }
}

// Verify device locally
function verifyLocalDevice(deviceId, password) {
    const device = localAuthorizedDevices.get(deviceId);
    if (!device) {
        return { success: false, error: 'Unknown device' };
    }
    
    const hashedPassword = hashPassword(password);
    if (device.password !== hashedPassword) {
        return { success: false, error: 'Invalid password' };
    }
    
    return { success: true, deviceName: device.name, source: 'local' };
}

// Main verification function
async function verifyDevice(deviceId, password) {
    if (!REQUIRE_AUTH) return { success: true };
    
    // Try CrowdCount API first if enabled
    if (USE_CROWDCOUNT_AUTH) {
        const crowdCountResult = await verifyCrowdCountDevice(deviceId, password);
        if (crowdCountResult.success || !FALLBACK_TO_LOCAL) {
            return crowdCountResult;
        }
    }
    
    // Try local authentication
    return verifyLocalDevice(deviceId, password);
}

// Make CrowdCount API call with stored credentials
async function makeCrowdCountCall(endpoint, params, connection) {
    if (!connection.deviceId || !connection.password) {
        throw new Error('No credentials stored for this connection');
    }
    
    try {
        const response = await axios.get(`${CROWDCOUNT_API_URL}/${endpoint}`, {
            params: {
                user_id: connection.deviceId,
                password: connection.password,
                ...params
            },
            timeout: 5000
        });
        return response.data;
    } catch (error) {
        console.error(`CrowdCount API error (${endpoint}):`, error.message);
        throw error;
    }
}

// Middleware to check admin credentials
function requireAdmin(req, res, next) {
    const authHeader = req.headers.authorization;
    
    if (!app.locals.adminUsername || !app.locals.adminPassword) {
        return res.status(501).json({ error: 'Admin interface not configured' });
    }
    
    if (!authHeader || !authHeader.startsWith('Basic ')) {
        return res.status(401).json({ error: 'Authentication required' });
    }
    
    const base64Credentials = authHeader.split(' ')[1];
    const credentials = Buffer.from(base64Credentials, 'base64').toString('ascii');
    const [username, password] = credentials.split(':');
    
    if (username !== app.locals.adminUsername || hashPassword(password) !== app.locals.adminPassword) {
        return res.status(401).json({ error: 'Invalid credentials' });
    }
    
    next();
}

// Basic HTTP endpoints
app.use(express.json());

app.get('/health', (req, res) => {
    const authenticatedConnections = Array.from(connections.values())
        .filter(conn => conn.authenticated).length;
    
    res.json({ 
        status: 'healthy', 
        connections: connections.size,
        authenticated: authenticatedConnections,
        feeds: feeds.size,
        authRequired: REQUIRE_AUTH,
        crowdCountAuth: USE_CROWDCOUNT_AUTH,
        crowdCountAPI: CROWDCOUNT_API_URL,
        localDevicesConfigured: localAuthorizedDevices.size
    });
});

app.get('/feeds', (req, res) => {
    const feedList = Array.from(feeds.entries()).map(([id, feed]) => ({
        id,
        name: feed.name,
        deviceId: feed.deviceId,
        viewers: feed.viewers.size
    }));
    res.json(feedList);
});

// Get current count from CrowdCount for a device
app.get('/device/:deviceId/count', async (req, res) => {
    const { deviceId } = req.params;
    const connection = Array.from(connections.values()).find(c => c.deviceId === deviceId);
    
    if (!connection) {
        return res.status(404).json({ error: 'Device not connected' });
    }
    
    try {
        const data = await makeCrowdCountCall('currentcount', {}, connection);
        res.json(data);
    } catch (error) {
        res.status(500).json({ error: 'Failed to get count from CrowdCount' });
    }
});

// List authorized devices (admin only)
app.get('/devices', requireAdmin, (req, res) => {
    const devices = Array.from(localAuthorizedDevices.entries()).map(([id, device]) => ({
        id,
        name: device.name,
        source: 'local'
    }));
    
    // Add connected devices from CrowdCount
    const connectedDevices = Array.from(connections.values())
        .filter(conn => conn.authenticated && conn.authSource === 'crowdcount')
        .map(conn => ({
            id: conn.deviceId,
            name: conn.deviceId,
            source: 'crowdcount',
            online: true
        }));
    
    res.json({ 
        local: devices,
        crowdcount: connectedDevices,
        total: devices.length + connectedDevices.length
    });
});

// Add/update local device (admin only or if registration is allowed)
app.post('/devices', (req, res) => {
    if (!ALLOW_DEVICE_REGISTRATION) {
        requireAdmin(req, res, () => {
            registerLocalDevice(req, res);
        });
    } else {
        registerLocalDevice(req, res);
    }
});

function registerLocalDevice(req, res) {
    const { deviceId, password, name } = req.body;
    
    if (!deviceId || !password) {
        return res.status(400).json({ error: 'Device ID and password required' });
    }
    
    localAuthorizedDevices.set(deviceId, {
        password: hashPassword(password),
        name: name || deviceId
    });
    
    console.log(`Local device registered: ${deviceId} (${name || deviceId})`);
    
    res.json({ 
        message: 'Device registered locally',
        deviceId,
        name: name || deviceId
    });
}

// Delete local device (admin only)
app.delete('/devices/:deviceId', requireAdmin, (req, res) => {
    const { deviceId } = req.params;
    
    if (localAuthorizedDevices.has(deviceId)) {
        localAuthorizedDevices.delete(deviceId);
        
        // Disconnect any active connections for this device
        connections.forEach((conn, clientId) => {
            if (conn.deviceId === deviceId) {
                conn.ws.close();
            }
        });
        
        res.json({ message: 'Device removed', deviceId });
    } else {
        res.status(404).json({ error: 'Device not found' });
    }
});

// WebSocket handling
wss.on('connection', (ws) => {
    const clientId = crypto.randomBytes(16).toString('hex');
    
    connections.set(clientId, { 
        ws, 
        type: null, 
        feedId: null,
        deviceId: null,
        password: null, // Store password for later API calls
        authenticated: !REQUIRE_AUTH, // Auto-auth if auth is disabled
        authSource: null // 'crowdcount' or 'local'
    });
    
    console.log(`Client connected: ${clientId}`);
    
    // Send initial message requesting authentication
    if (REQUIRE_AUTH) {
        ws.send(JSON.stringify({
            type: 'auth-required',
            clientId,
            message: 'Please authenticate with device ID and password',
            authMethod: USE_CROWDCOUNT_AUTH ? 'crowdcount' : 'local'
        }));
        
        // Set authentication timeout
        const authTimeout = setTimeout(() => {
            const connection = connections.get(clientId);
            if (connection && !connection.authenticated) {
                console.log(`Client ${clientId} failed to authenticate in time`);
                ws.send(JSON.stringify({
                    type: 'auth-timeout',
                    message: 'Authentication timeout'
                }));
                ws.close();
            }
        }, AUTH_TIMEOUT_MS);
        
        // Store timeout reference for cleanup
        connections.get(clientId).authTimeout = authTimeout;
    } else {
        // No auth required, send connected message immediately
        ws.send(JSON.stringify({
            type: 'connected',
            clientId,
            iceServers: [{ urls: 'stun:stun.l.google.com:19302' }]
        }));
    }
    
    ws.on('message', (message) => {
        try {
            const data = JSON.parse(message);
            handleMessage(clientId, data);
        } catch (err) {
            console.error('Error parsing message:', err);
        }
    });
    
    ws.on('close', () => {
        console.log(`Client disconnected: ${clientId}`);
        handleDisconnect(clientId);
    });
});

async function handleMessage(clientId, data) {
    const connection = connections.get(clientId);
    if (!connection) return;
    
    // Handle authentication first
    if (data.type === 'authenticate') {
        await handleAuthentication(clientId, data);
        return;
    }
    
    // Check if authenticated for all other message types
    if (!connection.authenticated) {
        connection.ws.send(JSON.stringify({
            type: 'error',
            message: 'Authentication required'
        }));
        return;
    }
    
    switch(data.type) {
        case 'register-sender':
            // Register as video sender
            connection.type = 'sender';
            connection.feedId = data.feedId || clientId;
            
            feeds.set(connection.feedId, {
                senderId: clientId,
                name: data.name || `Feed ${connection.feedId}`,
                deviceId: connection.deviceId,
                viewers: new Set()
            });
            
            connection.ws.send(JSON.stringify({
                type: 'registered',
                feedId: connection.feedId
            }));
            
            // Update device status in CrowdCount if using their auth
            if (connection.authSource === 'crowdcount') {
                try {
                    await updateCrowdCountStatus(connection, 'online');
                } catch (error) {
                    console.error('Failed to update CrowdCount status:', error.message);
                }
            }
            
            // Notify all viewers of new feed
            broadcastToViewers({ 
                type: 'feed-added', 
                feedId: connection.feedId,
                name: data.name,
                deviceId: connection.deviceId
            });
            
            console.log(`Sender registered: ${clientId}, feed: ${connection.feedId}, device: ${connection.deviceId}`);
            break;
            
        case 'register-viewer':
            // Register as viewer
            connection.type = 'viewer';
            
            const availableFeeds = Array.from(feeds.entries()).map(([id, feed]) => ({
                id,
                name: feed.name,
                deviceId: feed.deviceId,
                viewers: feed.viewers.size
            }));
            
            connection.ws.send(JSON.stringify({
                type: 'registered',
                feeds: availableFeeds
            }));
            console.log(`Viewer registered: ${clientId}`);
            break;
            
        case 'request-feed':
            // Viewer wants to watch a feed
            const feed = feeds.get(data.feedId);
            if (!feed) {
                connection.ws.send(JSON.stringify({
                    type: 'error',
                    message: 'Feed not found'
                }));
                return;
            }
            
            feed.viewers.add(clientId);
            
            // Notify sender about new viewer
            const sender = connections.get(feed.senderId);
            if (sender) {
                sender.ws.send(JSON.stringify({
                    type: 'viewer-joined',
                    viewerId: clientId,
                    feedId: data.feedId
                }));
            }
            console.log(`Viewer ${clientId} joined feed ${data.feedId}`);
            break;
            
        case 'offer':
        case 'answer':
        case 'ice-candidate':
            // Forward WebRTC signaling messages
            const target = connections.get(data.targetId);
            if (target && target.ws.readyState === WebSocket.OPEN) {
                target.ws.send(JSON.stringify({
                    type: data.type,
                    senderId: clientId,
                    data: data.data || data.offer || data.answer || data.candidate
                }));
            }
            break;
    }
}

async function handleAuthentication(clientId, data) {
    const connection = connections.get(clientId);
    if (!connection) return;
    
    const { deviceId, password } = data;
    
    if (!deviceId || !password) {
        connection.ws.send(JSON.stringify({
            type: 'auth-failed',
            message: 'Device ID and password required'
        }));
        return;
    }
    
    const verification = await verifyDevice(deviceId, password);
    
    if (verification.success) {
        // Clear auth timeout
        if (connection.authTimeout) {
            clearTimeout(connection.authTimeout);
            delete connection.authTimeout;
        }
        
        // Mark as authenticated and store credentials
        connection.authenticated = true;
        connection.deviceId = deviceId;
        connection.password = password; // Store password for future CrowdCount API calls
        connection.authSource = verification.source || 'unknown';
        
        console.log(`Client ${clientId} authenticated as device: ${deviceId} (via ${connection.authSource})`);
        
        // Send success response with connection info
        connection.ws.send(JSON.stringify({
            type: 'auth-success',
            clientId,
            deviceId,
            deviceName: verification.deviceName,
            authSource: connection.authSource,
            iceServers: [{ urls: 'stun:stun.l.google.com:19302' }]
        }));
        
        // Also send connected message for backward compatibility
        connection.ws.send(JSON.stringify({
            type: 'connected',
            clientId,
            iceServers: [{ urls: 'stun:stun.l.google.com:19302' }]
        }));
        
    } else {
        console.log(`Authentication failed for client ${clientId}: ${verification.error}`);
        connection.ws.send(JSON.stringify({
            type: 'auth-failed',
            message: verification.error
        }));
    }
}

async function updateCrowdCountStatus(connection, status) {
    if (!connection.deviceId || !connection.password) {
        return;
    }
    
    try {
        await axios.post(`${CROWDCOUNT_API_URL}/update_status`, 
            new URLSearchParams({
                user_id: connection.deviceId,
                password: connection.password,
                status: status
            }),
            {
                headers: { 'Content-Type': 'application/x-www-form-urlencoded' },
                timeout: 5000
            }
        );
        console.log(`Updated CrowdCount status for ${connection.deviceId}: ${status}`);
    } catch (error) {
        console.error(`Failed to update CrowdCount status: ${error.message}`);
    }
}

async function handleDisconnect(clientId) {
    const connection = connections.get(clientId);
    if (!connection) return;
    
    // Clear auth timeout if exists
    if (connection.authTimeout) {
        clearTimeout(connection.authTimeout);
    }
    
    // Update status in CrowdCount if authenticated via their API
    if (connection.authSource === 'crowdcount' && connection.authenticated) {
        try {
            await updateCrowdCountStatus(connection, 'offline');
        } catch (error) {
            console.error('Failed to update CrowdCount status on disconnect:', error.message);
        }
    }
    
    if (connection.type === 'sender' && connection.feedId) {
        // Remove feed and notify viewers
        const feed = feeds.get(connection.feedId);
        if (feed) {
            feed.viewers.forEach(viewerId => {
                const viewer = connections.get(viewerId);
                if (viewer) {
                    viewer.ws.send(JSON.stringify({
                        type: 'feed-ended',
                        feedId: connection.feedId
                    }));
                }
            });
            feeds.delete(connection.feedId);
            broadcastToViewers({ 
                type: 'feed-removed', 
                feedId: connection.feedId 
            });
        }
    }
    
    connections.delete(clientId);
}

function broadcastToViewers(message) {
    connections.forEach(conn => {
        if (conn.type === 'viewer' && conn.authenticated && conn.ws.readyState === WebSocket.OPEN) {
            conn.ws.send(JSON.stringify(message));
        }
    });
}

// Start server
server.listen(PORT, () => {
    console.log(`Video streaming server running on port ${PORT}`);
    console.log(`HTTP: http://localhost:${PORT}`);
    console.log(`WebSocket: ws://localhost:${PORT}`);
    console.log(`Authentication: ${REQUIRE_AUTH ? 'ENABLED' : 'DISABLED'}`);
    
    if (REQUIRE_AUTH) {
        if (USE_CROWDCOUNT_AUTH) {
            console.log(`Using CrowdCount API for authentication: ${CROWDCOUNT_API_URL}`);
            if (FALLBACK_TO_LOCAL) {
                console.log(`Fallback to local auth: ENABLED (${localDeviceCount} local devices)`);
            }
        } else {
            console.log(`Using local authentication only: ${localDeviceCount} devices configured`);
        }
    }
    
    if (ALLOW_DEVICE_REGISTRATION) {
        console.log('Device registration: OPEN (devices can self-register locally)');
    } else if (app.locals.adminUsername) {
        console.log('Device registration: ADMIN ONLY');
    } else {
        console.log('Device registration: DISABLED');
    }
});