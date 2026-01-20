export default function Loading() {
  return (
    <div className="p-6 space-y-6">
      {/* Header skeleton */}
      <div className="flex items-center justify-between">
        <div className="space-y-2">
          <div className="h-8 w-48 bg-secondary rounded animate-pulse" />
          <div className="h-4 w-64 bg-secondary/50 rounded animate-pulse" />
        </div>
        <div className="flex items-center gap-3">
          <div className="h-10 w-40 bg-secondary rounded animate-pulse" />
          <div className="h-10 w-28 bg-secondary rounded animate-pulse" />
        </div>
      </div>

      {/* Statistics skeleton */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
        {[1, 2, 3, 4].map((i) => (
          <div key={i} className="p-4 bg-card border border-border rounded-lg">
            <div className="flex items-center gap-3">
              <div className="w-10 h-10 bg-secondary rounded-lg animate-pulse" />
              <div className="space-y-2 flex-1">
                <div className="h-3 w-20 bg-secondary/50 rounded animate-pulse" />
                <div className="h-6 w-16 bg-secondary rounded animate-pulse" />
              </div>
            </div>
          </div>
        ))}
      </div>

      {/* Filters skeleton */}
      <div className="p-4 bg-card border border-border rounded-lg">
        <div className="flex items-center gap-3">
          <div className="h-10 flex-1 bg-secondary rounded animate-pulse" />
          <div className="h-10 w-48 bg-secondary rounded animate-pulse" />
          <div className="h-10 w-48 bg-secondary rounded animate-pulse" />
        </div>
      </div>

      {/* Grid skeleton */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
        {[1, 2, 3, 4, 5, 6].map((i) => (
          <div key={i} className="bg-card border border-border rounded-lg overflow-hidden">
            <div className="aspect-video bg-secondary animate-pulse" />
            <div className="p-3 space-y-2">
              <div className="h-4 w-3/4 bg-secondary rounded animate-pulse" />
              <div className="h-3 w-1/2 bg-secondary/50 rounded animate-pulse" />
              <div className="h-3 w-full bg-secondary/50 rounded animate-pulse" />
            </div>
          </div>
        ))}
      </div>
    </div>
  )
}
