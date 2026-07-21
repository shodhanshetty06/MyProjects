import { motion } from "framer-motion";
import { Building2, FileText, Pin, PinOff, Search } from "lucide-react";
import { useMemo, useState } from "react";

import { staggerChildren } from "@/components/motion/variants";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/Card";
import { EmptyState } from "@/components/ui/EmptyState";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/Tabs";
import { type PinnedItemType, usePinsStore } from "@/stores/pinsStore";

const TYPE_META: Record<PinnedItemType, { label: string; icon: typeof Search }> = {
  job: { label: "Jobs", icon: Search },
  resume: { label: "Resumes", icon: FileText },
  company: { label: "Companies", icon: Building2 },
};

export function PinnedItemsPanel() {
  const [active, setActive] = useState<PinnedItemType>("job");
  const allItems = usePinsStore((state) => state.items);
  const unpin = usePinsStore((state) => state.unpin);
  const items = useMemo(() => allItems.filter((item) => item.type === active), [allItems, active]);
  const Icon = TYPE_META[active].icon;

  return (
    <Card>
      <CardHeader className="flex-row items-center justify-between gap-3 space-y-0">
        <CardTitle className="flex items-center gap-1.5">
          <Pin className="size-4 text-brand" aria-hidden />
          Pinned
        </CardTitle>
        <Tabs value={active} onValueChange={(v) => setActive(v as PinnedItemType)}>
          <TabsList>
            {(Object.keys(TYPE_META) as PinnedItemType[]).map((type) => (
              <TabsTrigger key={type} value={type} className="px-2.5 py-1 text-xs">
                {TYPE_META[type].label}
              </TabsTrigger>
            ))}
          </TabsList>
        </Tabs>
      </CardHeader>
      <CardContent>
        <Tabs value={active}>
          <TabsContent value={active} forceMount>
            {items.length === 0 ? (
              <EmptyState icon={Icon} title={`No pinned ${TYPE_META[active].label.toLowerCase()}`} description="Pin items as you browse to keep them handy here." />
            ) : (
              <motion.ul variants={staggerChildren(0.05)} initial="hidden" animate="visible" className="space-y-1.5">
                {items.map((item) => (
                  <motion.li
                    key={item.id}
                    variants={{ hidden: { opacity: 0 }, visible: { opacity: 1 } }}
                    className="flex items-center gap-2.5 rounded-[var(--radius-md)] border border-border px-2.5 py-2"
                  >
                    <div className="min-w-0 flex-1">
                      <p className="truncate text-sm font-medium text-ink">{item.label}</p>
                      {item.meta && <p className="truncate text-xs text-ink-muted">{item.meta}</p>}
                    </div>
                    <button
                      type="button"
                      onClick={() => unpin(item.type, item.id)}
                      className="shrink-0 rounded-[var(--radius-sm)] p-1 text-ink-muted transition-colors hover:bg-surface-2 hover:text-ink"
                      aria-label={`Unpin ${item.label}`}
                    >
                      <PinOff className="size-3.5" aria-hidden />
                    </button>
                  </motion.li>
                ))}
              </motion.ul>
            )}
          </TabsContent>
        </Tabs>
      </CardContent>
    </Card>
  );
}
