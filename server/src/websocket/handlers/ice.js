// server/src/websocket/handlers/ice.js - ICE candidate handlers
const logger = require('../../utils/logger');
const connectionManager = require('../../services/connectionManager');
const wsServer = require('../wsServer');

function handleIceCandidate(clientId, data) {
    const { targetId, candidate, feedId } = data;
    
    if (!targetId || !candidate) {
        wsServer.sendToClient(clientId, {
            type: 'error',
            message: 'Missing targetId or candidate',
            code: 'INVALID_ICE_CANDIDATE'
        });
        return;
    }
    
    const target = connectionManager.getConnection(targetId);
    
    if (!target) {
        logger.warn(`ICE candidate target ${targetId} not found`);
        // Don't send error back as this is common during disconnections
        return;
    }
    
    logger.debug(`Forwarding ICE candidate from ${clientId} to ${targetId}`);
    
    wsServer.sendToClient(targetId, {
        type: 'ice-candidate',
        candidate: candidate,
        senderId: clientId,
        feedId: feedId
    });
}

function handleIceRestart(clientId, data) {
    const { targetId, feedId } = data;
    
    if (!targetId) {
        wsServer.sendToClient(clientId, {
            type: 'error',
            message: 'Missing targetId',
            code: 'INVALID_ICE_RESTART'
        });
        return;
    }
    
    const target = connectionManager.getConnection(targetId);
    
    if (!target) {
        wsServer.sendToClient(clientId, {
            type: 'error',
            message: 'Target client not found',
            code: 'TARGET_NOT_FOUND'
        });
        return;
    }
    
    logger.info(`ICE restart requested from ${clientId} to ${targetId}`);
    
    wsServer.sendToClient(targetId, {
        type: 'ice-restart',
        senderId: clientId,
        feedId: feedId
    });
}

module.exports = {
    handleIceCandidate,
    handleIceRestart
};