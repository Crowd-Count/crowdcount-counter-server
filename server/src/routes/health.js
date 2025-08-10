// server/src/routes/health.js - Health check endpoints
const express = require('express');
const router = express.Router();
const config = require('../config');
const connectionManager = require('../services/connectionManager');
const feedManager = require('../services/feedManager');
const logger = require('../utils/logger');

// Basic health check
router.get('/', (req, res) => {
    res.status(200).json({
        status: 'healthy',
        timestamp: new Date().toISOString(),
        uptime: process.uptime(),
        environment: config.nodeEnv,
        version: config.version
    });
});

// Detailed health check
router.get('/detailed', (req, res) => {
    const memoryUsage = process.memoryUsage();
    const connectionStats = connectionManager.getConnectionStats();
    const feedStats = feedManager.getFeedStats();
    
    const health = {
        status: 'healthy',
        timestamp: new Date().toISOString(),
        server: {
            uptime: process.uptime(),
            environment: config.nodeEnv,
            version: config.version,
            nodeVersion: process.version,
            pid: process.pid
        },
        memory: {
            rss: `${Math.round(memoryUsage.rss / 1024 / 1024)} MB`,
            heapTotal: `${Math.round(memoryUsage.heapTotal / 1024 / 1024)} MB`,
            heapUsed: `${Math.round(memoryUsage.heapUsed / 1024 / 1024)} MB`,
            external: `${Math.round(memoryUsage.external / 1024 / 1024)} MB`
        },
        connections: {
            total: connectionStats.total,
            byType: connectionStats.byType,
            averageConnectionTime: `${Math.round(connectionStats.averageConnectionTime)} seconds`
        },
        feeds: {
            total: feedStats.totalFeeds,
            totalViewers: feedStats.totalViewers,
            byCodec: feedStats.byCodec,
            byResolution: feedStats.byResolution
        },
        system: {
            platform: process.platform,
            cpus: require('os').cpus().length,
            totalMemory: `${Math.round(require('os').totalmem() / 1024 / 1024)} MB`,
            freeMemory: `${Math.round(require('os').freemem() / 1024 / 1024)} MB`,
            loadAverage: require('os').loadavg()
        }
    };
    
    // Check for warning conditions
    const warnings = [];
    
    if (connectionStats.total > config.websocket.maxConnections * 0.8) {
        warnings.push('High connection count');
    }
    
    if (memoryUsage.heapUsed / memoryUsage.heapTotal > 0.9) {
        warnings.push('High memory usage');
    }
    
    if (warnings.length > 0) {
        health.status = 'warning';
        health.warnings = warnings;
    }
    
    res.status(200).json(health);
});

// Liveness probe (for Kubernetes)
router.get('/live', (req, res) => {
    res.status(200).json({
        status: 'alive',
        timestamp: new Date().toISOString()
    });
});

// Readiness probe (for Kubernetes)
router.get('/ready', async (req, res) => {
    try {
        // Check if services are ready
        const checks = {
            websocket: true, // Always true if server is running
            connections: connectionManager.getAllConnections().length >= 0,
            feeds: feedManager.getAllFeeds().length >= 0
        };
        
        // Check Redis if enabled
        if (config.redis.enabled) {
            try {
                const Redis = require('ioredis');
                const redis = new Redis(config.redis.url);
                await redis.ping();
                checks.redis = true;
                redis.disconnect();
            } catch (error) {
                checks.redis = false;
                logger.error('Redis health check failed:', error);
            }
        }
        
        const allHealthy = Object.values(checks).every(v => v === true);
        
        if (allHealthy) {
            res.status(200).json({
                status: 'ready',
                timestamp: new Date().toISOString(),
                checks
            });
        } else {
            res.status(503).json({
                status: 'not ready',
                timestamp: new Date().toISOString(),
                checks
            });
        }
    } catch (error) {
        logger.error('Readiness check failed:', error);
        res.status(503).json({
            status: 'error',
            message: 'Readiness check failed',
            timestamp: new Date().toISOString()
        });
    }
});

module.exports = router;