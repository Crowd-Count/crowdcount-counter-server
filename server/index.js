// server/index.js - Main entry point
require('dotenv').config();
const config = require('./src/config');
const logger = require('./src/utils/logger');
const app = require('./src/app');
const wsServer = require('./src/websocket/wsServer');

const PORT = config.port || 3000;

// Create HTTP/HTTPS server based on environment
let server;
if (config.isProduction && config.ssl.enabled) {
    const https = require('https');
    const fs = require('fs');
    
    const credentials = {
        key: fs.readFileSync(config.ssl.keyPath, 'utf8'),
        cert: fs.readFileSync(config.ssl.certPath, 'utf8')
    };
    
    server = https.createServer(credentials, app);
    logger.info('HTTPS server created');
} else {
    const http = require('http');
    server = http.createServer(app);
    logger.info('HTTP server created');
}

// Initialize WebSocket server
wsServer.initialize(server);

// Start server
server.listen(PORT, () => {
    logger.info(`Server running on port ${PORT}`);
    logger.info(`Environment: ${config.nodeEnv}`);
    logger.info(`WebSocket endpoint: ${config.isProduction ? 'wss' : 'ws'}://localhost:${PORT}`);
});

// Graceful shutdown
process.on('SIGTERM', () => {
    logger.info('SIGTERM received, closing server...');
    
    wsServer.close(() => {
        server.close(() => {
            logger.info('Server closed');
            process.exit(0);
        });
    });
});

process.on('uncaughtException', (error) => {
    logger.error('Uncaught Exception:', error);
    process.exit(1);
});

process.on('unhandledRejection', (reason, promise) => {
    logger.error('Unhandled Rejection at:', promise, 'reason:', reason);
    process.exit(1);
});