// server/src/websocket/handlers/signaling.js - WebRTC signaling handlers
const logger = require('../../utils/logger');
const connectionManager = require('../../services/connectionManager');
const wsServer = require('../wsServer');

function handleOffer(clientId, data) {
    const { targetId, offer, feedId } = data;
    
    if (!targetId || !offer) {
        wsServer.sendToClient(clientId, {
            type: 'error',
            message: 'Missing targetId or offer',
            code: 'INVALID_OFFER'
        });
        return;
    }
    
    const sender = connectionManager.getConnection(clientId);
    const target = connectionManager.getConnection(targetId);
    
    if (!target) {
        wsServer.sendToClient(clientId, {
            type: 'error',
            message: 'Target client not found',
            code: 'TARGET_NOT_FOUND'
        });
        return;
    }
    
    logger.info(`Forwarding offer from ${clientId} to ${targetId}`);
    
    wsServer.sendToClient(targetId, {
        type: 'offer',
        offer: offer,
        senderId: clientId,
        feedId: feedId || sender.feedId,
        senderInfo: {
            id: clientId,
            type: sender.type
        }
    });
}

function handleAnswer(clientId, data) {
    const { targetId, answer } = data;
    
    if (!targetId || !answer) {
        wsServer.sendToClient(clientId, {
            type: 'error',
            message: 'Missing targetId or answer',
            code: 'INVALID_ANSWER'
        });
        return;
    }
    
    const viewer = connectionManager.getConnection(clientId);
    const target = connectionManager.getConnection(targetId);
    
    if (!target) {
        wsServer.sendToClient(clientId, {
            type: 'error',
            message: 'Target client not found',
            code: 'TARGET_NOT_FOUND'
        });
        return;
    }
    
    logger.info(`Forwarding answer from ${clientId} to ${targetId}`);
    
    wsServer.sendToClient(targetId, {
        type: 'answer',
        answer: answer,
        viewerId: clientId,
        viewerInfo: {
            id: clientId,
            type: viewer.type
        }
    });
}

module.exports = {
    handleOffer,
    handleAnswer
};