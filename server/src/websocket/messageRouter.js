// server/src/websocket/messageRouter.js - Routes WebSocket messages to appropriate handlers
const logger = require('../utils/logger');
const signalingHandler = require('./handlers/signaling');
const feedHandler = require('./handlers/feed');
const iceHandler = require('./handlers/ice');

const handlers = {
    // Feed management
    'register-sender': feedHandler.registerSender,
    'register-viewer': feedHandler.registerViewer,
    'request-feed': feedHandler.requestFeed,
    'stop-feed': feedHandler.stopFeed,
    'update-feed': feedHandler.updateFeed,
    'list-feeds': feedHandler.listFeeds,
    
    // WebRTC signaling
    'offer': signalingHandler.handleOffer,
    'answer': signalingHandler.handleAnswer,
    'ice-candidate': iceHandler.handleIceCandidate,
    'ice-restart': iceHandler.handleIceRestart,
    
    // Connection management
    'ping': (clientId) => {
        const wsServer = require('./wsServer');
        wsServer.sendToClient(clientId, { type: 'pong', timestamp: Date.now() });
    },
    
    // Stats and monitoring
    'get-stats': (clientId, data) => {
        const connectionManager = require('../services/connectionManager');
        const feedManager = require('../services/feedManager');
        const wsServer = require('./wsServer');
        
        wsServer.sendToClient(clientId, {
            type: 'stats',
            data: {
                connections: connectionManager.getConnectionStats(),
                feeds: feedManager.getFeedStats(),
                timestamp: new Date().toISOString()
            }
        });
    }
};

function route(clientId, message) {
    const { type, ...data } = message;
    
    if (!type) {
        logger.warn(`Message from ${clientId} missing type field`);
        return;
    }
    
    const handler = handlers[type];
    
    if (!handler) {
        logger.warn(`Unknown message type from ${clientId}: ${type}`);
        const wsServer = require('./wsServer');
        wsServer.sendToClient(clientId, {
            type: 'error',
            message: `Unknown message type: ${type}`,
            code: 'UNKNOWN_MESSAGE_TYPE'
        });
        return;
    }
    
    try {
        logger.debug(`Routing ${type} message from ${clientId}`);
        handler(clientId, data);
    } catch (error) {
        logger.error(`Error handling ${type} message from ${clientId}:`, error);
        const wsServer = require('./wsServer');
        wsServer.sendToClient(clientId, {
            type: 'error',
            message: 'Internal server error',
            code: 'HANDLER_ERROR'
        });
    }
}

module.exports = {
    route
};