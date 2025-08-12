// simple-video-server.js - Minimal WebRTC signaling server for multiple video feeds
const express = require('express');
const WebSocket = require('ws');
const http = require('http');

const app = express();
const server = http.createServer(app);
const wss = new WebSocket.Server({ server });

// Store active connections and feeds
const connections = new Map();  // clientId -> {ws, type, feedId}
const feeds = new Map();        // feedId -> {senderId, name, viewers: Set}

// Basic HTTP endpoints
app.use(express.json());

app.get('/health', (req, res) => {
    res.json({ 
        status: 'healthy', 
        connections: connections.size,
        feeds: feeds.size 
    });
});

app.get('/feeds', (req, res) => {
    const feedList = Array.from(feeds.entries()).map(([id, feed]) => ({
        id,
        name: feed.name,
        viewers: feed.viewers.size
    }));
    res.json(feedList);
});

// WebSocket handling
wss.on('connection', (ws) => {
    const clientId = Math.random().toString(36).substring(7);
    
    connections.set(clientId, { ws, type: null, feedId: null });
    console.log(`Client connected: ${clientId}`);
    
    // Send welcome message
    ws.send(JSON.stringify({
        type: 'connected',
        clientId,
        iceServers: [{ urls: 'stun:stun.l.google.com:19302' }]
    }));
    
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

function handleMessage(clientId, data) {
    const connection = connections.get(clientId);
    if (!connection) return;
    
    switch(data.type) {
        case 'register-sender':
            // Register as video sender
            connection.type = 'sender';
            connection.feedId = data.feedId || clientId;
            
            feeds.set(connection.feedId, {
                senderId: clientId,
                name: data.name || `Feed ${connection.feedId}`,
                viewers: new Set()
            });
            
            connection.ws.send(JSON.stringify({
                type: 'registered',
                feedId: connection.feedId
            }));
            
            // Notify all viewers of new feed
            broadcastToViewers({ type: 'feed-added', feedId: connection.feedId });
            console.log(`Sender registered: ${clientId}, feed: ${connection.feedId}`);
            break;
            
        case 'register-viewer':
            // Register as viewer
            connection.type = 'viewer';
            
            const availableFeeds = Array.from(feeds.keys());
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

function handleDisconnect(clientId) {
    const connection = connections.get(clientId);
    if (!connection) return;
    
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
            broadcastToViewers({ type: 'feed-removed', feedId: connection.feedId });
        }
    }
    
    connections.delete(clientId);
}

function broadcastToViewers(message) {
    connections.forEach(conn => {
        if (conn.type === 'viewer' && conn.ws.readyState === WebSocket.OPEN) {
            conn.ws.send(JSON.stringify(message));
        }
    });
}

// Start server
const PORT = process.env.PORT || 3000;
server.listen(PORT, () => {
    console.log(`Video streaming server running on port ${PORT}`);
    console.log(`HTTP: http://localhost:${PORT}`);
    console.log(`WebSocket: ws://localhost:${PORT}`);
});