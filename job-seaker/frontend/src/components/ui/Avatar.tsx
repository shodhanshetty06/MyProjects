import { cn } from "@/lib/cn";

interface AvatarProps {
  name: string;
  src?: string | null;
  size?: "sm" | "md" | "lg";
  className?: string;
}

const SIZE_CLASSES = {
  sm: "size-6 text-[10px]",
  md: "size-8 text-xs",
  lg: "size-11 text-sm",
};

function initialsFrom(name: string): string {
  const parts = name.trim().split(/[\s@.]+/).filter(Boolean);
  if (parts.length === 0) return "?";
  if (parts.length === 1) return parts[0].slice(0, 2).toUpperCase();
  return (parts[0][0] + parts[1][0]).toUpperCase();
}

export function Avatar({ name, src, size = "md", className }: AvatarProps) {
  if (src) {
    return (
      <img
        src={src}
        alt={name}
        className={cn("rounded-full object-cover", SIZE_CLASSES[size], className)}
      />
    );
  }
  return (
    <div
      role="img"
      aria-label={name}
      className={cn(
        "flex shrink-0 items-center justify-center rounded-full font-semibold text-white",
        "bg-[linear-gradient(135deg,var(--color-accent-a),var(--color-accent-b))]",
        SIZE_CLASSES[size],
        className,
      )}
    >
      {initialsFrom(name)}
    </div>
  );
}
