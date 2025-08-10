// server/src/routes/stats.js - Server statistics endpoints
const express = require('express');
const router = express.Router();
const os = require('os');
const { asyncHandler } = require('../middleware/errorHandler');
const connectionManager = require('../services/connectionManager');
const feedManager = require('../services/feedManager');
const wsServer = require('../websocket/wsServer');

// Get overall server statistics
router.get('/', asyncHandler(async (req, res) => {
    const stats = wsServer.getStats();
    
    res.json({
        success: true,
        stats: {
            server: {
                uptime: process.uptime(),
                timestamp: new Date().toISOString(),
                version: require('../../package.json').version
            },
            connections: stats.connections,
            feeds: feedManager.getFeedStats(),
            resources: {
                memory: stats.memory,
                cpu: process.cpuUsage()
            }
        }
    });
}));

// Get connection statistics
router.get('/connections', asyncHandler(async (req, res) => {
    const connections = connectionManager.getAllConnections();
    const stats = connectionManager.getConnectionStats();
    
    res.json({
        success: true,
        stats: {
            total: stats.total,
            byType: stats.byType,
            averageConnectionTime: stats.averageConnectionTime,
            byIP: stats.byIP,
            connections: connections.map(conn => ({
                id: conn.id,
                type: conn.type,
                ip: conn.ip,
                connectedAt: conn.timestamp,
                feedId: conn.feedId,
                userAgent: conn.userAgent
            }))
        }
    });
}));

// Get system metrics
router.get('/system', asyncHandler(async (req, res) => {
    const memoryUsage = process.memoryUsage();
    const cpuUsage = process.cpuUsage();
    
    res.json({
        success: true,
        system: {
            process: {
                pid: process.pid,
                version: process.version,
                uptime: process.uptime(),
                memory: {
                    rss: memoryUsage.rss,
                    heapTotal: memoryUsage.heapTotal,
                    heapUsed: memoryUsage.heapUsed,
                    external: memoryUsage.external,
                    arrayBuffers: memoryUsage.arrayBuffers
                },
                cpu: {
                    user: cpuUsage.user,
                    system: cpuUsage.system
                }
            },
            os: {
                platform: os.platform(),
                arch: os.arch(),
                version: os.version(),
                release: os.release(),
                hostname: os.hostname(),
                cpus: os.cpus().map(cpu => ({
                    model: cpu.model,
                    speed: cpu.speed,
                    times: cpu.times
                })),
                memory: {
                    total: os.totalmem(),
                    free: os.freemem(),
                    used: os.totalmem() - os.freemem()
                },
                loadAverage: os.loadavg(),
                uptime: os.uptime(),
                networkInterfaces: Object.entries(os.networkInterfaces()).map(([name, interfaces]) => ({
                    name,
                    interfaces: interfaces.map(int => ({
                        address: int.address,
                        family: int.family,
                        internal: int.internal
                    }))
                }))
            }
        },
        timestamp: new Date().toISOString()
    });
}));

// Get bandwidth statistics (estimated)
router.get('/bandwidth', asyncHandler(async (req, res) => {
    const feeds = feedManager.getAllFeeds();
    
    let totalBandwidth = 0;
    const bandwidthByFeed = feeds.map(feed => {
        const bandwidth = feed.bitrate * feed.viewers.size;
        totalBandwidth += bandwidth;
        
        return {
            feedId: feed.id,
            feedName: feed.name,
            bitrate: feed.bitrate,
            viewers: feed.viewers.size,
            estimatedBandwidth: bandwidth,
            estimatedBandwidthMbps: (bandwidth / 1000000).toFixed(2)
        };
    });
    
    res.json({
        success: true,
        bandwidth: {
            total: totalBandwidth,
            totalMbps: (totalBandwidth / 1000000).toFixed(2),
            totalGbps: (totalBandwidth / 1000000000).toFixed(3),
            byFeed: bandwidthByFeed,
            timestamp: new Date().toISOString()
        }
    });
}));

// Get historical metrics (if stored)
router.get('/history', asyncHandler(async (req, res) => {
    // This would typically query a time-series database
    // For now, return current snapshot
    res.json({
        success: true,
        message: 'Historical metrics not yet implemented',
        currentSnapshot: {
            connections: connectionManager.getConnectionStats(),
            feeds: feedManager.getFeedStats(),
            timestamp: new Date().toISOString()
        }
    });
}));

module.exports = router;