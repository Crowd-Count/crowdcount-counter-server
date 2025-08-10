// server/src/middleware/errorHandler.js - Global error handler
const logger = require('../utils/logger');
const config = require('../config');

class AppError extends Error {
    constructor(message, statusCode, code = null) {
        super(message);
        this.statusCode = statusCode;
        this.code = code;
        this.isOperational = true;
        Error.captureStackTrace(this, this.constructor);
    }
}

const errorHandler = (err, req, res, next) => {
    let error = { ...err };
    error.message = err.message;
    
    // Log error
    logger.error('Error Handler', {
        error: {
            message: error.message,
            statusCode: error.statusCode,
            code: error.code,
            stack: err.stack
        },
        request: {
            method: req.method,
            url: req.url,
            ip: req.ip,
            body: req.body
        }
    });
    
    // Mongoose bad ObjectId
    if (err.name === 'CastError') {
        const message = 'Invalid ID format';
        error = new AppError(message, 400, 'INVALID_ID');
    }
    
    // Mongoose duplicate key
    if (err.code === 11000) {
        const message = 'Duplicate field value';
        error = new AppError(message, 400, 'DUPLICATE_VALUE');
    }
    
    // Mongoose validation error
    if (err.name === 'ValidationError') {
        const message = Object.values(err.errors).map(val => val.message).join(', ');
        error = new AppError(message, 400, 'VALIDATION_ERROR');
    }
    
    // JWT errors
    if (err.name === 'JsonWebTokenError') {
        const message = 'Invalid token';
        error = new AppError(message, 401, 'INVALID_TOKEN');
    }
    
    if (err.name === 'TokenExpiredError') {
        const message = 'Token expired';
        error = new AppError(message, 401, 'TOKEN_EXPIRED');
    }
    
    // Default error
    const statusCode = error.statusCode || 500;
    const message = error.message || 'Internal Server Error';
    const code = error.code || 'INTERNAL_ERROR';
    
    res.status(statusCode).json({
        success: false,
        error: {
            message,
            code,
            ...(config.isDevelopment && { 
                stack: err.stack,
                details: error 
            })
        }
    });
};

// Async error wrapper
const asyncHandler = (fn) => (req, res, next) => {
    Promise.resolve(fn(req, res, next)).catch(next);
};

// Not found handler
const notFound = (req, res, next) => {
    const error = new AppError(`Not found - ${req.originalUrl}`, 404, 'NOT_FOUND');
    next(error);
};

module.exports = {
    errorHandler,
    AppError,
    asyncHandler,
    notFound
};