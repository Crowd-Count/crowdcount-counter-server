// server/src/routes/feeds.js - Feed management API endpoints
const express = require('express');
const router = express.Router();
const { asyncHandler, AppError } = require('../middleware/errorHandler');
const feedManager = require('../services/feedManager');
const connectionManager = require('../services/connectionManager');
const logger = require('../utils/logger');

// Get all active feeds
router.get('/', asyncHandler(async (req, res) => {
    const feeds = feedManager.getAllFeeds();
    
    const feedList = feeds.map(feed => ({
        id: feed.id,
        name: feed.name,
        senderId: feed.clientId,
        resolution: feed.resolution,
        frameRate: feed.frameRate,
        codec: feed.codec,
        bitrate: feed.bitrate,
        viewerCount: feed.viewers.size,
        createdAt: feed.timestamp,
        stats: {
            totalViewers: feed.stats.totalViewers,
            peakViewers: feed.stats.peakViewers,
            duration: Math.round((Date.now() - feed.stats.startTime) / 1000)
        }
    }));
    
    res.json({
        success: true,
        count: feedList.length,
        feeds: feedList
    });
}));

// Get specific feed details
router.get('/:feedId', asyncHandler(async (req, res) => {
    const { feedId } = req.params;
    const feed = feedManager.getFeed(feedId);
    
    if (!feed) {
        throw new AppError('Feed not found', 404, 'FEED_NOT_FOUND');
    }
    
    const sender = connectionManager.getConnection(feed.clientId);
    
    res.json({
        success: true,
        feed: {
            id: feed.id,
            name: feed.name,
            senderId: feed.clientId,
            senderInfo: sender ? {
                ip: sender.ip,
                connectedAt: sender.timestamp,
                userAgent: sender.userAgent
            } : null,
            resolution: feed.resolution,
            frameRate: feed.frameRate,
            codec: feed.codec,
            bitrate: feed.bitrate,
            metadata: feed.metadata,
            viewers: {
                current: feed.viewers.size,
                list: Array.from(feed.viewers),
                total: feed.stats.totalViewers,
                peak: feed.stats.peakViewers
            },
            createdAt: feed.timestamp,
            duration: Math.round((Date.now() - feed.stats.startTime) / 1000)
        }
    });
}));

// Get feed statistics
router.get('/:feedId/stats', asyncHandler(async (req, res) => {
    const { feedId } = req.params;
    const feed = feedManager.getFeed(feedId);
    
    if (!feed) {
        throw new AppError('Feed not found', 404, 'FEED_NOT_FOUND');
    }
    
    const now = Date.now();
    const duration = (now - feed.stats.startTime) / 1000;
    
    res.json({
        success: true,
        stats: {
            feedId: feed.id,
            name: feed.name,
            duration: Math.round(duration),
            viewers: {
                current: feed.viewers.size,
                total: feed.stats.totalViewers,
                peak: feed.stats.peakViewers,
                average: feed.stats.totalViewers > 0 ? 
                    Math.round(duration / feed.stats.totalViewers) : 0
            },
            bandwidth: {
                estimatedCurrent: feed.bitrate * feed.viewers.size,
                estimatedTotal: feed.bitrate * feed.stats.totalViewers * duration
            },
            quality: {
                resolution: feed.resolution,
                frameRate: feed.frameRate,
                codec: feed.codec,
                bitrate: feed.bitrate
            }
        }
    });
}));

// Get viewers of a feed
router.get('/:feedId/viewers', asyncHandler(async (req, res) => {
    const { feedId } = req.params;
    const feed = feedManager.getFeed(feedId);
    
    if (!feed) {
        throw new AppError('Feed not found', 404, 'FEED_NOT_FOUND');
    }
    
    const viewers = Array.from(feed.viewers).map(viewerId => {
        const connection = connectionManager.getConnection(viewerId);
        return {
            id: viewerId,
            connectedAt: connection ? connection.timestamp : null,
            ip: connection ? connection.ip : null
        };
    });
    
    res.json({
        success: true,
        feedId: feed.id,
        count: viewers.length,
        viewers: viewers
    });
}));

// Stop a feed (requires authentication in production)
router.delete('/:feedId', asyncHandler(async (req, res) => {
    const { feedId } = req.params;
    const { reason = 'API request' } = req.body;
    
    const feed = feedManager.getFeed(feedId);
    
    if (!feed) {
        throw new AppError('Feed not found', 404, 'FEED_NOT_FOUND');
    }
    
    // In production, verify authorization here
    // For now, we'll just log the action
    logger.warn(`Feed ${feedId} stop requested via API`, {
        feedId,
        reason,
        ip: req.ip
    });
    
    // Notify viewers
    const wsServer = require('../websocket/wsServer');
    feed.viewers.forEach(viewerId => {
        wsServer.sendToClient(viewerId, {
            type: 'feed-stopped',
            feedId: feedId,
            reason: reason
        });
    });
    
    // Notify sender
    wsServer.sendToClient(feed.clientId, {
        type: 'feed-stop-requested',
        feedId: feedId,
        reason: reason
    });
    
    // Remove feed
    const removedFeed = feedManager.removeFeed(feedId);
    
    res.json({
        success: true,
        message: 'Feed stopped successfully',
        feed: {
            id: removedFeed.id,
            name: removedFeed.name,
            finalStats: removedFeed.stats
        }
    });
}));

// Get feed summary statistics
router.get('/stats/summary', asyncHandler(async (req, res) => {
    const stats = feedManager.getFeedStats();
    
    res.json({
        success: true,
        summary: {
            totalFeeds: stats.totalFeeds,
            totalViewers: stats.totalViewers,
            byCodec: stats.byCodec,
            byResolution: stats.byResolution,
            averageViewersPerFeed: stats.totalFeeds > 0 ? 
                Math.round(stats.totalViewers / stats.totalFeeds * 10) / 10 : 0
        },
        timestamp: new Date().toISOString()
    });
}));

module.exports = router;