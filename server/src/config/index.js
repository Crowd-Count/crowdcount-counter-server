// server/src/config/index.js - Main configuration file
const path = require('path');

const config = {
    // Environment
    nodeEnv: process.env.NODE_ENV || 'development',
    isProduction: process.env.NODE_ENV === 'production',
    isDevelopment: process.env.NODE_ENV === 'development',
    isTest: process.env.NODE_ENV === 'test',
    
    // Server
    port: parseInt(process.env.PORT, 10) || 3000,
    host: process.env.HOST || '0.0.0.0',
    version: process.env.npm_package_version || '1.0.0',
    
    // SSL Configuration
    ssl: {
        enabled: process.env.SSL_ENABLED === 'true',
        keyPath: process.env.SSL_KEY_PATH || path.join(__dirname, '../../../certs/private-key.pem'),
        certPath: process.env.SSL_CERT_PATH || path.join(__dirname, '../../../certs/certificate.pem'),
    },
    
    // CORS
    cors: {
        origins: process.env.CORS_ORIGINS ? 
            process.env.CORS_ORIGINS.split(',') : 
            ['http://localhost:3000', 'http://localhost:3001'],
    },
    
    // WebSocket
    websocket: {
        pingInterval: parseInt(process.env.WS_PING_INTERVAL, 10) || 30000,
        pingTimeout: parseInt(process.env.WS_PING_TIMEOUT, 10) || 5000,
        maxConnections: parseInt(process.env.WS_MAX_CONNECTIONS, 10) || 1000,
        maxFeedsPerSender: parseInt(process.env.MAX_FEEDS_PER_SENDER, 10) || 5,
    },
    
    // Redis (for scaling)
    redis: {
        enabled: process.env.REDIS_ENABLED === 'true',
        url: process.env.REDIS_URL || 'redis://localhost:6379',
        password: process.env.REDIS_PASSWORD,
        db: parseInt(process.env.REDIS_DB, 10) || 0,
    },
    
    // Database (optional)
    database: {
        enabled: process.env.DATABASE_ENABLED === 'true',
        url: process.env.DATABASE_URL,
        pool: {
            min: parseInt(process.env.DB_POOL_MIN, 10) || 2,
            max: parseInt(process.env.DB_POOL_MAX, 10) || 10,
        },
    },
    
    // AWS Configuration
    aws: {
        region: process.env.AWS_REGION || 'us-east-1',
        accessKeyId: process.env.AWS_ACCESS_KEY_ID,
        secretAccessKey: process.env.AWS_SECRET_ACCESS_KEY,
        s3: {
            bucket: process.env.AWS_S3_BUCKET,
            recordingsPrefix: process.env.AWS_S3_RECORDINGS_PREFIX || 'recordings/',
        },
        cloudwatch: {
            enabled: process.env.CLOUDWATCH_ENABLED === 'true',
            namespace: process.env.CLOUDWATCH_NAMESPACE || 'VideoStreaming',
        },
    },
    
    // Authentication
    auth: {
        enabled: process.env.AUTH_ENABLED === 'true',
        jwtSecret: process.env.JWT_SECRET || 'change-this-secret-in-production',
        jwtExpiry: process.env.JWT_EXPIRY || '24h',
        apiKeys: process.env.API_KEYS ? process.env.API_KEYS.split(',') : [],
    },
    
    // Rate Limiting
    rateLimit: {
        windowMs: parseInt(process.env.RATE_LIMIT_WINDOW_MS, 10) || 15 * 60 * 1000, // 15 minutes
        max: parseInt(process.env.RATE_LIMIT_MAX, 10) || 100,
        message: 'Too many requests from this IP, please try again later.',
    },
    
    // Logging
    logging: {
        level: process.env.LOG_LEVEL || 'info',
        format: process.env.LOG_FORMAT || 'json',
        directory: process.env.LOG_DIRECTORY || path.join(__dirname, '../../../logs'),
        maxFiles: parseInt(process.env.LOG_MAX_FILES, 10) || 14,
        maxSize: process.env.LOG_MAX_SIZE || '20m',
    },
    
    // TURN/STUN servers for WebRTC
    ice: {
        servers: [
            {
                urls: process.env.STUN_SERVER || 'stun:stun.l.google.com:19302',
            },
            ...(process.env.TURN_SERVER ? [{
                urls: process.env.TURN_SERVER,
                username: process.env.TURN_USERNAME,
                credential: process.env.TURN_PASSWORD,
            }] : []),
        ],
    },
    
    // Static files
    serveStatic: process.env.SERVE_STATIC === 'true',
    
    // Monitoring
    monitoring: {
        healthCheckInterval: parseInt(process.env.HEALTH_CHECK_INTERVAL, 10) || 30000,
        metricsEnabled: process.env.METRICS_ENABLED === 'true',
        metricsPort: parseInt(process.env.METRICS_PORT, 10) || 9090,
    },
};

// Validate required configuration in production
if (config.isProduction) {
    const required = ['JWT_SECRET', 'AWS_ACCESS_KEY_ID', 'AWS_SECRET_ACCESS_KEY'];
    const missing = required.filter(key => !process.env[key]);
    
    if (missing.length > 0) {
        throw new Error(`Missing required environment variables: ${missing.join(', ')}`);
    }
}

module.exports = config;