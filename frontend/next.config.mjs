/** @type {import('next').NextConfig} */
const nextConfig = {
  // Silence workspace root warning
  outputFileTracingRoot: '.',
  
  // Proxy API requests to backend during development
  async rewrites() {
    const backendUrl = process.env.NEXT_PUBLIC_BACKEND_URL || 'http://localhost:8080';
    
    return [
      {
        source: '/api/:path*',
        destination: `${backendUrl}/:path*`,
      },
      {
        source: '/record/:path*',
        destination: `${backendUrl}/record/:path*`,
      },
    ];
  },
  
  // WebSocket proxy is handled separately by the client
  // Enable experimental features if needed
  experimental: {
    serverActions: {
      bodySizeLimit: '2mb',
    },
  },
  
  // Image optimization
  images: {
    remotePatterns: [
      {
        protocol: 'http',
        hostname: 'localhost',
        port: '8080',
        pathname: '/record/**',
      },
      {
        protocol: 'https',
        hostname: 'images.unsplash.com',
      },
    ],
  },
};

export default nextConfig;
