// server/src/middleware/rateLimiter.js - Rate limiting middleware
const rateLimit = require('express-rate-limit');
const RedisStore = require('rate-limit-redis');
const Redis = require('ioredis');
const config = require('../config');
const logger = require('../utils/logger');

// Create rate limiter based on configuration
let limiterOptions = {
    windowMs: config.rateLimit.windowMs,
    max: config.rateLimit.max,
    message: config.rateLimit.message,
    standardHeaders: true, // Return rate limit info in headers
    legacyHeaders: false,
    handler: (req, res) => {
        logger.warn('Rate limit exceeded', {
            ip: req.ip,
            path: req.path,
            method: req.method
        });
        
        res.status(429).json({
            success: false,
            error: {
                message: config.rateLimit.message,
                code: 'RATE_LIMIT_EXCEEDED',
                retryAfter: req.rateLimit.resetTime
            }
        });
    },
    skip: (req) => {
        // Skip rate limiting for health checks
        return req.path === '/health';
    }
};

// Use Redis store if Redis is enabled
if (config.redis.enabled) {
    try {
        const redisClient = new Redis({
            host: config.redis.url,
            password: config.redis.password,
            db: config.redis.db,
            enableOfflineQueue: false
        });
        
        redisClient.on('error', (err) => {
            logger.error('Redis error in rate limiter:', err);
        });
        
        limiterOptions.store = new RedisStore({
            client: redisClient,
            prefix: 'rl:',
        });
        
        logger.info('Rate limiter using Redis store');
    } catch (error) {
        logger.error('Failed to initialize Redis for rate limiting:', error);
        logger.info('Falling back to memory store for rate limiting');
    }
} else {
    logger.info('Rate limiter using memory store');
}

// Create different limiters for different endpoints
const createLimiter = (options = {}) => {
    return rateLimit({
        ...limiterOptions,
        ...options
    });
};

// Default API limiter
const apiLimiter = createLimiter();

// Strict limiter for authentication endpoints
const authLimiter = createLimiter({
    windowMs: 15 * 60 * 1000, // 15 minutes
    max: 5, // 5 requests per window
    message: 'Too many authentication attempts, please try again later',
    skipSuccessfulRequests: true
});

// WebSocket connection limiter
const wsLimiter = createLimiter({
    windowMs: 1 * 60 * 1000, // 1 minute
    max: 10, // 10 connections per minute
    message: 'Too many WebSocket connection attempts'
});

// File upload limiter
const uploadLimiter = createLimiter({
    windowMs: 15 * 60 * 1000, // 15 minutes
    max: 10, // 10 uploads per window
    message: 'Too many file uploads, please try again later'
});

module.exports = {
    apiLimiter