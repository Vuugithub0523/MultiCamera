/**
 * Logger utility
 */

import winston from 'winston';

const logFormat = winston.format.printf(({ level, message, label, timestamp }) => {
  return `${timestamp} | ${level.toUpperCase().padEnd(8)} | ${label} | ${message}`;
});

export function createLogger(label: string): winston.Logger {
  return winston.createLogger({
    level: process.env.LOG_LEVEL || 'info',
    format: winston.format.combine(
      winston.format.label({ label }),
      winston.format.timestamp({ format: 'YYYY-MM-DD HH:mm:ss' }),
      logFormat
    ),
    transports: [
      new winston.transports.Console(),
    ],
  });
}
