// server/src/services/feedManager.js - Manages video feeds
const logger = require('../utils/logger');
const EventEmitter = require('events');

class FeedManager extends EventEmitter {
    constructor() {
        super();
        this.feeds = new Map();
        this.feedsBySender = new Map(); // clientId -> Set of feedIds
    }
    
    createFeed(feedData) {
        const feed = {
            id: feedData.id,
            name: feedData.name,
            clientId: feedData.clientId,
            resolution: feedData.resolution,
            frameRate: feedData.frameRate,
            codec: feedData.codec,
            bitrate: feedData.bitrate,
            metadata: feedData.metadata || {},
            viewers: new Set(),
            timestamp: new Date().toISOString(),
            stats: {
                totalViewers: 0,
                peakViewers: 0,
                totalDuration: 0,
                startTime: Date.now()
            }
        };
        
        this.feeds.set(feed.id, feed);
        
        // Track feeds by sender
        if (!this.feedsBySender.has(feedData.clientId)) {
            this.feedsBySender.set(feedData.clientId, new Set());
        }
        this.feedsBySender.get(feedData.clientId).add(feed.id);
        
        this.emit('feed:created', feed);
        logger.info(`Feed created: ${feed.id} by ${feed.clientId}`);
        
        return feed;
    }
    
    updateFeed(feedId, updates) {
        const feed = this.feeds.get(feedId);
        if (!feed) {
            logger.warn(`Attempted to update non-existent feed: ${feedId}`);
            return null;
        }
        
        // Only update allowed fields
        const allowedUpdates = ['name', 'resolution', 'frameRate', 'codec', 'bitrate', 'metadata'];
        const filteredUpdates = {};
        
        allowedUpdates.forEach(field => {
            if (updates[field] !== undefined) {
                filteredUpdates[field] = updates[field];
            }
        });
        
        Object.assign(feed, filteredUpdates);
        feed.lastUpdated = new Date().toISOString();
        
        this.emit('feed:updated', { feedId, updates: filteredUpdates });
        logger.info(`Feed updated: ${feedId}`);
        
        return feed;
    }
    
    removeFeed(feedId) {
        const feed = this.feeds.get(feedId);
        if (!feed) {
            logger.warn(`Attempted to remove non-existent feed: ${feedId}`);
            return null;
        }
        
        // Update stats
        feed.stats.totalDuration = Date.now() - feed.stats.startTime;
        
        // Remove from sender's feed list
        const senderFeeds = this.feedsBySender.get(feed.clientId);
        if (senderFeeds) {
            senderFeeds.delete(feedId);
            if (senderFeeds.size === 0) {
                this.feedsBySender.delete(feed.clientId);
            }
        }
        
        this.feeds.delete(feedId);
        
        this.emit('feed:removed', feed);
        logger.info(`Feed removed: ${feedId}, had ${feed.viewers.size} viewers`);
        
        return feed;
    }
    
    getFeed(feedId) {
        return this.feeds.get(feedId);
    }
    
    getAllFeeds() {
        return Array.from(this.feeds.values());
    }
    
    getFeedsBySender(clientId) {
        const feedIds = this.feedsBySender.get(clientId);
        if (!feedIds) return [];
        
        return Array.from(feedIds).map(id => this.feeds.get(id)).filter(Boolean);
    }
    
    addViewer(feedId, viewerId) {
        const feed = this.feeds.get(feedId);
        if (!feed) {
            logger.warn(`Attempted to add viewer to non-existent feed: ${feedId}`);
            return false;
        }
        
        feed.viewers.add(viewerId);
        feed.stats.totalViewers++;
        
        // Update peak viewers
        if (feed.viewers.size > feed.stats.peakViewers) {
            feed.stats.peakViewers = feed.viewers.size;
        }
        
        this.emit('viewer:joined', { feedId, viewerId, viewerCount: feed.viewers.size });
        logger.debug(`Viewer ${viewerId} joined feed ${feedId}`);
        
        return true;
    }
    
    removeViewer(feedId, viewerId) {
        const feed = this.feeds.get(feedId);
        if (!feed) return false;
        
        const removed = feed.viewers.delete(viewerId);
        
        if (removed) {
            this.emit('viewer:left', { feedId, viewerId, viewerCount: feed.viewers.size });
            logger.debug(`Viewer ${viewerId} left feed ${feedId}`);
        }
        
        return removed;
    }
    
    removeViewerFromAllFeeds(viewerId) {
        const affectedFeeds = [];
        
        this.feeds.forEach((feed, feedId) => {
            if (feed.viewers.has(viewerId)) {
                feed.viewers.delete(viewerId);
                affectedFeeds.push(feedId);
            }
        });
        
        if (affectedFeeds.length > 0) {
            logger.debug(`Removed viewer ${viewerId} from ${affectedFeeds.length} feeds`);
        }
        
        return affectedFeeds;
    }
    
    handleSenderDisconnect(clientId, feedId) {
        // Remove all feeds from this sender
        const senderFeeds = this.getFeedsBySender(clientId);
        
        senderFeeds.forEach(feed => {
            // Notify viewers
            const wsServer = require('../websocket/wsServer');
            feed.viewers.forEach(viewerId => {
                wsServer.sendToClient(viewerId, {
                    type: 'feed-disconnected',
                    feedId: feed.id,
                    reason: 'Sender disconnected'
                });
            });
            
            // Remove feed
            this.removeFeed(feed.id);
        });
        
        logger.info(`Removed ${senderFeeds.length} feeds from disconnected sender ${clientId}`);
    }
    
    getFeedStats() {
        const feeds = this.getAllFeeds();
        const now = Date.now();
        
        return {
            totalFeeds: feeds.length,
            totalViewers: feeds.reduce((sum, feed) => sum + feed.viewers.size, 0),
            feedDetails: feeds.map(feed => ({
                id: feed.id,
                name: feed.name,
                senderId: feed.clientId,
                viewerCount: feed.viewers.size,
                duration: (now - feed.stats.startTime) / 1000, // in seconds
                peakViewers: feed.stats.peakViewers,
                totalViewers: feed.stats.totalViewers
            })),
            byCodec: this.groupFeedsByProperty('codec'),
            byResolution: this.groupFeedsByProperty('resolution')
        };
    }
    
    groupFeedsByProperty(property) {
        const groups = {};
        this.feeds.forEach(feed => {
            const value = feed[property];
            if (!groups[value]) {
                groups[value] = 0;
            }
            groups[value]++;
        });
        return groups;
    }
    
    cleanup() {
        // Remove feeds with no active sender connection
        const connectionManager = require('./connectionManager');
        const removed = [];
        
        this.feeds.forEach((feed, feedId) => {
            if (!connectionManager.isConnected(feed.clientId)) {
                this.handleSenderDisconnect(feed.clientId, feedId);
                removed.push(feedId);
            }
        });
        
        return removed.length;
    }
}

// Singleton instance
const feedManager = new FeedManager();

// Periodic cleanup
setInterval(() => {
    const cleaned = feedManager.cleanup();
    if (cleaned > 0) {
        logger.info(`Cleaned up ${cleaned} orphaned feeds`);
    }
}, 60000); // Every minute

module.exports = feedManager;