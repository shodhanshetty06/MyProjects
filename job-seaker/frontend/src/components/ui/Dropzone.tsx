import { motion } from "framer-motion";
import { UploadCloud } from "lucide-react";
import { useRef, useState, type DragEvent } from "react";

import { cn } from "@/lib/cn";

interface DropzoneProps {
  onFile: (file: File) => void;
  accept?: string;
  label?: string;
  hint?: string;
  disabled?: boolean;
  className?: string;
}

export function Dropzone({ onFile, accept = ".pdf,.docx", label = "Drop your resume here", hint = "PDF or DOCX, up to 10MB", disabled, className }: DropzoneProps) {
  const [isDragging, setIsDragging] = useState(false);
  const inputRef = useRef<HTMLInputElement>(null);

  function handleDrop(e: DragEvent<HTMLDivElement>) {
    e.preventDefault();
    setIsDragging(false);
    if (disabled) return;
    const file = e.dataTransfer.files?.[0];
    if (file) onFile(file);
  }

  return (
    <motion.div
      onDragOver={(e) => {
        e.preventDefault();
        if (!disabled) setIsDragging(true);
      }}
      onDragLeave={() => setIsDragging(false)}
      onDrop={handleDrop}
      onClick={() => !disabled && inputRef.current?.click()}
      animate={{ scale: isDragging ? 1.01 : 1 }}
      className={cn(
        "flex cursor-pointer flex-col items-center justify-center gap-2 rounded-[var(--radius-lg)] border-2 border-dashed px-6 py-10 text-center transition-colors",
        isDragging ? "border-brand bg-brand-soft" : "border-border hover:border-border-strong hover:bg-surface-2",
        disabled && "cursor-not-allowed opacity-50",
        className,
      )}
    >
      <div className="flex size-11 items-center justify-center rounded-full bg-brand-soft text-brand">
        <UploadCloud className="size-5" aria-hidden />
      </div>
      <p className="text-sm font-medium text-ink">{label}</p>
      <p className="text-xs text-ink-muted">{hint}</p>
      <input
        ref={inputRef}
        type="file"
        accept={accept}
        className="hidden"
        disabled={disabled}
        onChange={(e) => {
          const file = e.target.files?.[0];
          if (file) onFile(file);
          e.target.value = "";
        }}
      />
    </motion.div>
  );
}
