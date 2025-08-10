// server/src/websocket/wsServer.js - WebSocket server setup
const WebSocket = require('ws');
const config = require('../config');
const logger = require('../utils/logger');
const messageRouter = require('./messageRouter');
const connectionManager = require('../services/connectionManager');

let wss = null;

function initialize(server) {
    wss = new WebSocket.Server({ 
        server,
        maxPayload: 10 * 1024 * 1024, // 10MB max message size
        perMessageDeflate: {
            zlibDeflateOptions: {
                chunkSize: 1024,
                memLevel: 7,
                level: 3
            },
            zlibInflateOptions: {
                chunkSize: 10 * 1024
            },
            clientNoContextTakeover: true,
            serverNoContextTakeover: true,
            serverMaxWindowBits: 10,
            concurrencyLimit: 10,
            threshold: 1024
        }
    });
    
    wss.on('connection', handleConnection);
    
    // Set up ping/pong for connection health checks
    const interval = setInterval(() => {
        wss.clients.forEach((ws) => {
            if (ws.isAlive === false) {
                logger.warn(`Terminating dead connection: ${ws.clientId}`);
                return ws.terminate();
            }
            
            ws.isAlive = false;
            ws.ping();
        });
    }, config.websocket.pingInterval);
    
    wss.on('close', () => {
        clearInterval(interval);
    });
    
    logger.info('WebSocket server initialized');
    return wss;
}

function handleConnection(ws, request) {
    const clientIp = request.socket.remoteAddress;
    const clientId = connectionManager.generateClientId();
    
    // Check connection limits
    if (wss.clients.size > config.websocket.maxConnections) {
        ws.send(JSON.stringify({
            type: 'error',
            message: 'Server at maximum capacity',
            code: 'MAX_CONNECTIONS'
        }));
        ws.close(1008, 'Max connections reached');
        return;
    }
    
    // Set up connection properties
    ws.clientId = clientId;
    ws.clientIp = clientIp;
    ws.isAlive = true;
    ws.joinedAt = new Date();
    
    // Register connection
    connectionManager.addConnection(clientId, {
        ws,
        id: clientId,
        ip: clientIp,
        timestamp: new Date().toISOString(),
        type: null,
        feedId: null,
        userAgent: request.headers['user-agent']
    });
    
    logger.info(`New WebSocket connection: ${clientId} from ${clientIp}`);
    
    // Send welcome message
    ws.send(JSON.stringify({
        type: 'connection',
        clientId: clientId,
        message: 'Connected to signaling server',
        serverTime: new Date().toISOString(),
        iceServers: config.ice.servers
    }));
    
    // Set up event handlers
    ws.on('message', (message) => {
        try {
            const data = JSON.parse(message);
            messageRouter.route(clientId, data);
        } catch (error) {
            logger.error(`Error parsing message from ${clientId}:`, error);
            ws.send(JSON.stringify({
                type: 'error',
                message: 'Invalid message format',
                code: 'PARSE_ERROR'
            }));
        }
    });
    
    ws.on('pong', () => {
        ws.isAlive = true;
    });
    
    ws.on('close', (code, reason) => {
        logger.info(`Client disconnected: ${clientId}, code: ${code}, reason: ${reason}`);
        connectionManager.removeConnection(clientId);
    });
    
    ws.on('error', (error) => {
        logger.error(`WebSocket error for ${clientId}:`, error);
    });
}

function broadcast(message, filter = null) {
    if (!wss) return;
    
    const messageStr = JSON.stringify(message);
    wss.clients.forEach((client) => {
        if (client.readyState === WebSocket.OPEN) {
            if (!filter || filter(client)) {
                client.send(messageStr);
            }
        }
    });
}

function sendToClient(clientId, message) {
    const connection = connectionManager.getConnection(clientId);
    if (connection && connection.ws.readyState === WebSocket.OPEN) {
        connection.ws.send(JSON.stringify(message));
        return true;
    }
    return false;
}

function close(callback) {
    if (!wss) {
        if (callback) callback();
        return;
    }
    
    // Notify all clients
    broadcast({
        type: 'server-shutdown',
        message: 'Server is shutting down'
    });
    
    // Close all connections
    wss.clients.forEach((client) => {
        client.close(1001, 'Server shutdown');
    });
    
    wss.close(callback);
}

function getStats() {
    if (!wss) return null;
    
    return {
        totalConnections: wss.clients.size,
        connections: connectionManager.getConnectionStats(),
        uptime: process.uptime(),
        memory: process.memoryUsage()
    };
}

module.exports = {
    initialize,
    broadcast,
    sendToClient,
    close,
    getStats
};