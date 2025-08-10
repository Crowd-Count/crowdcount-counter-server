// server/src/websocket/handlers/feed.js - Feed management handlers
const logger = require('../../utils/logger');
const connectionManager = require('../../services/connectionManager');
const feedManager = require('../../services/feedManager');
const wsServer = require('../wsServer');
const config = require('../../config');

function registerSender(clientId, data) {
    const connection = connectionManager.getConnection(clientId);
    if (!connection) return;
    
    // Check if sender already has max feeds
    const existingFeeds = feedManager.getFeedsBySender(clientId);
    if (existingFeeds.length >= config.websocket.maxFeedsPerSender) {
        wsServer.sendToClient(clientId, {
            type: 'error',
            message: `Maximum feeds per sender reached (${config.websocket.maxFeedsPerSender})`,
            code: 'MAX_FEEDS_REACHED'
        });
        return;
    }
    
    const feedId = data.feedId || `${clientId}-${Date.now()}`;
    
    // Update connection type
    connectionManager.updateConnection(clientId, {
        type: 'sender',
        feedId: feedId
    });
    
    // Create feed
    const feed = feedManager.createFeed({
        id: feedId,
        name: data.feedName || `Feed ${feedId}`,
        clientId: clientId,
        resolution: data.resolution || '1920x1080',
        frameRate: data.frameRate || 30,
        codec: data.codec || 'h264',
        bitrate: data.bitrate || 2500000,
        metadata: data.metadata || {}
    });
    
    logger.info(`Registered sender ${clientId} with feed ${feedId}`);
    
    wsServer.sendToClient(clientId, {
        type: 'registered',
        feedId: feedId,
        message: 'Registered as video sender',
        feed: feed
    });
    
    // Notify all viewers of new feed
    broadcastFeedUpdate();
}

function registerViewer(clientId, data) {
    const connection = connectionManager.getConnection(clientId);
    if (!connection) return;
    
    // Update connection type
    connectionManager.updateConnection(clientId, {
        type: 'viewer'
    });
    
    const availableFeeds = feedManager.getAllFeeds();
    
    logger.info(`Registered viewer ${clientId}`);
    
    wsServer.sendToClient(clientId, {
        type: 'registered',
        message: 'Registered as viewer',
        availableFeeds: availableFeeds.map(feed => ({
            id: feed.id,
            name: feed.name,
            resolution: feed.resolution,
            frameRate: feed.frameRate,
            codec: feed.codec,
            viewerCount: feed.viewers.size,
            createdAt: feed.timestamp
        }))
    });
}

function requestFeed(clientId, data) {
    const { feedId } = data;
    
    if (!feedId) {
        wsServer.sendToClient(clientId, {
            type: 'error',
            message: 'Missing feedId',
            code: 'MISSING_FEED_ID'
        });
        return;
    }
    
    const feed = feedManager.getFeed(feedId);
    if (!feed) {
        wsServer.sendToClient(clientId, {
            type: 'error',
            message: 'Feed not found',
            code: 'FEED_NOT_FOUND'
        });
        return;
    }
    
    // Add viewer to feed
    feedManager.addViewer(feedId, clientId);
    
    // Notify sender about new viewer
    wsServer.sendToClient(feed.clientId, {
        type: 'viewer-request',
        viewerId: clientId,
        feedId: feedId,
        viewerCount: feed.viewers.size
    });
    
    // Send feed details to viewer
    wsServer.sendToClient(clientId, {
        type: 'feed-ready',
        feedId: feedId,
        senderId: feed.clientId,
        feedDetails: {
            name: feed.name,
            resolution: feed.resolution,
            frameRate: feed.frameRate,
            codec: feed.codec
        }
    });
    
    logger.info(`Viewer ${clientId} requested feed ${feedId}`);
}

function stopFeed(clientId, data) {
    const connection = connectionManager.getConnection(clientId);
    if (!connection || connection.type !== 'sender') {
        wsServer.sendToClient(clientId, {
            type: 'error',
            message: 'Only senders can stop feeds',
            code: 'UNAUTHORIZED'
        });
        return;
    }
    
    const feedId = data.feedId || connection.feedId;
    const feed = feedManager.getFeed(feedId);
    
    if (!feed) {
        wsServer.sendToClient(clientId, {
            type: 'error',
            message: 'Feed not found',
            code: 'FEED_NOT_FOUND'
        });
        return;
    }
    
    if (feed.clientId !== clientId) {
        wsServer.sendToClient(clientId, {
            type: 'error',
            message: 'Unauthorized to stop this feed',
            code: 'UNAUTHORIZED'
        });
        return;
    }
    
    // Notify all viewers
    feed.viewers.forEach(viewerId => {
        wsServer.sendToClient(viewerId, {
            type: 'feed-stopped',
            feedId: feedId,
            reason: data.reason || 'Sender stopped the feed'
        });
    });
    
    // Remove feed
    feedManager.removeFeed(feedId);
    
    // Update connection
    connectionManager.updateConnection(clientId, {
        feedId: null
    });
    
    logger.info(`Feed ${feedId} stopped by sender ${clientId}`);
    
    wsServer.sendToClient(clientId, {
        type: 'feed-stopped-confirmation',
        feedId: feedId
    });
    
    // Broadcast update to all viewers
    broadcastFeedUpdate();
}

function updateFeed(clientId, data) {
    const connection = connectionManager.getConnection(clientId);
    if (!connection || connection.type !== 'sender') {
        wsServer.sendToClient(clientId, {
            type: 'error',
            message: 'Only senders can update feeds',
            code: 'UNAUTHORIZED'
        });
        return;
    }
    
    const feedId = data.feedId || connection.feedId;
    const updates = {
        name: data.name,
        resolution: data.resolution,
        frameRate: data.frameRate,
        codec: data.codec
    };