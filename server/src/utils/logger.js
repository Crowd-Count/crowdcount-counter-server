// server/src/utils/logger.js - Logging utility
const winston = require('winston');
const path = require('path');
const fs = require('fs');
const config = require('../config');

// Ensure log directory exists
const logDir = config.logging.directory;
if (!fs.existsSync(logDir)) {
    fs.mkdirSync(logDir, { recursive: true });
}

// Define log format
const logFormat = winston.format.combine(
    winston.format.timestamp({ format: 'YYYY-MM-DD HH:mm:ss' }),
    winston.format.errors({ stack: true }),
    winston.format.splat(),
    winston.format.json()
);

// Console format for development
const consoleFormat = winston.format.combine(
    winston.format.colorize(),
    winston.format.timestamp({ format: 'YYYY-MM-DD HH:mm:ss' }),
    winston.format.printf(({ timestamp, level, message, ...meta }) => {
        let msg = `${timestamp} [${level}]: ${message}`;
        if (Object.keys(meta).length > 0) {
            msg += ` ${JSON.stringify(meta)}`;
        }
        return msg;
    })
);

// Create logger instance
const logger = winston.createLogger({
    level: config.logging.level,
    format: logFormat,
    defaultMeta: { 
        service: 'video-streaming-server',
        environment: config.nodeEnv
    },
    transports: [
        // Error logs
        new winston.transports.File({
            filename: path.join(logDir, 'error.log'),
            level: 'error',
            maxsize: config.logging.maxSize,
            maxFiles: config.logging.maxFiles,
            tailable: true
        }),
        // Combined logs
        new winston.transports.File({
            filename: path.join(logDir, 'combined.log'),
            maxsize: config.logging.maxSize,
            maxFiles: config.logging.maxFiles,
            tailable: true
        })
    ],
    exitOnError: false
});

// Add console transport for non-production environments
if (config.nodeEnv !== 'production') {
    logger.add(new winston.transports.Console({
        format: consoleFormat,
        handleExceptions: true
    }));
}

// Create a stream object for Morgan HTTP logging
logger.stream = {
    write: (message) => {
        logger.info(message.trim());
    }
};

// Helper methods for structured logging
logger.logRequest = (req, additionalInfo = {}) => {
    logger.info('HTTP Request', {
        method: req.method,
        url: req.url,
        ip: req.ip,
        userAgent: req.get('user-agent'),
        ...additionalInfo
    });
};

logger.logResponse = (req, res, additionalInfo = {}) => {
    logger.info('HTTP Response', {
        method: req.method,
        url: req.url,
        statusCode: res.statusCode,
        responseTime: res.responseTime,
        ...additionalInfo
    });
};

logger.logWebSocketEvent = (event, clientId, data = {}) => {
    logger.debug('WebSocket Event', {
        event,
        clientId,
        ...data
    });
};

logger.logError = (error, context = {}) => {
    logger.error('Error occurred', {
        message: error.message,
        stack: error.stack,
        ...context
    });
};

module.exports = logger;