import { DashboardLayout } from "@/components/dashboard-layout"
import { Skeleton } from "@/components/ui/skeleton"

export default function Loading() {
  return (
    <DashboardLayout>
      <div className="p-6 space-y-6">
        <div className="space-y-2">
          <Skeleton className="h-8 w-48" />
          <Skeleton className="h-4 w-64" />
        </div>
        <Skeleton className="h-10 w-96" />
        <div className="space-y-4">
          {Array.from({ length: 3 }).map((_, i) => (
            <Skeleton key={i} className="h-24" />
          ))}
        </div>
      </div>
    </DashboardLayout>
  )
}
