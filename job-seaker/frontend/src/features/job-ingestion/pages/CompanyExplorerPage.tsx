import { useQuery } from "@tanstack/react-query";
import { motion } from "framer-motion";
import { Building2, Search } from "lucide-react";
import { useMemo, useState } from "react";

import { PageHeader } from "@/components/layout/PageHeader";
import { staggerChildren } from "@/components/motion/variants";
import { EmptyState } from "@/components/ui/EmptyState";
import { ErrorState } from "@/components/ui/ErrorState";
import { Input } from "@/components/ui/Input";
import { Pagination } from "@/components/ui/Pagination";
import { Skeleton } from "@/components/ui/Skeleton";
import { listCompanies } from "@/features/job-ingestion/api";
import { CompanyCard } from "@/features/job-ingestion/components/CompanyCard";

export function CompanyExplorerPage() {
  const [page, setPage] = useState(1);
  const [query, setQuery] = useState("");

  const companiesQuery = useQuery({
    queryKey: ["companies", page],
    queryFn: () => listCompanies(page, 24),
  });

  const filtered = useMemo(() => {
    const companies = companiesQuery.data?.data ?? [];
    if (!query.trim()) return companies;
    const q = query.toLowerCase();
    return companies.filter((c) => c.name.toLowerCase().includes(q) || c.industry?.toLowerCase().includes(q));
  }, [companiesQuery.data, query]);

  return (
    <div>
      <PageHeader
        eyebrow="Job search"
        title="Company explorer"
        description="Browse companies with open roles on the platform."
      />

      <div className="mb-5 max-w-sm">
        <Input
          leadingIcon={<Search />}
          placeholder="Filter this page by name or industry"
          value={query}
          onChange={(e) => setQuery(e.target.value)}
        />
      </div>

      {companiesQuery.isLoading ? (
        <div className="grid gap-3 sm:grid-cols-2 lg:grid-cols-3">
          {Array.from({ length: 6 }).map((_, i) => (
            <Skeleton key={i} className="h-36 rounded-[var(--radius-lg)]" />
          ))}
        </div>
      ) : companiesQuery.isError ? (
        <ErrorState description="Couldn't load companies." />
      ) : filtered.length === 0 ? (
        <EmptyState icon={Building2} title="No companies found" description="Try clearing your filter." />
      ) : (
        <>
          <motion.div
            variants={staggerChildren(0.04)}
            initial="hidden"
            animate="visible"
            className="grid gap-3 sm:grid-cols-2 lg:grid-cols-3"
          >
            {filtered.map((company) => (
              <CompanyCard key={company.id} company={company} />
            ))}
          </motion.div>

          {companiesQuery.data && (
            <Pagination
              page={companiesQuery.data.pagination.page}
              totalPages={companiesQuery.data.pagination.total_pages}
              totalItems={companiesQuery.data.pagination.total_items}
              onPageChange={setPage}
            />
          )}
        </>
      )}
    </div>
  );
}
