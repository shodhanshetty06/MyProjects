import { useMutation, useQueryClient } from "@tanstack/react-query";
import { Upload } from "lucide-react";
import { useRef } from "react";

import { Button, type ButtonProps } from "@/components/ui/Button";
import { useAuth } from "@/features/auth";
import { uploadResume } from "@/features/resume-parser";
import { toast } from "@/stores/toastStore";

interface UploadResumeButtonProps extends Omit<ButtonProps, "onClick" | "isLoading"> {
  label?: string;
}

/**
 * Uploads a resume directly from wherever it's rendered (workspace cards),
 * instead of sending the user to the Resumes page just to pick a file.
 */
export function UploadResumeButton({ label = "Upload a resume", ...buttonProps }: UploadResumeButtonProps) {
  const { user } = useAuth();
  const userId = user!.id;
  const queryClient = useQueryClient();
  const inputRef = useRef<HTMLInputElement>(null);

  const upload = useMutation({
    mutationFn: (file: File) => uploadResume(userId, file),
    onSuccess: () => {
      toast.success("Resume uploaded", "We're parsing it now — this page will update automatically.");
      void queryClient.invalidateQueries({ queryKey: ["workspace"] });
    },
    onError: () => toast.danger("Upload failed", "Try a different file (PDF or DOCX)."),
  });

  return (
    <>
      <input
        ref={inputRef}
        type="file"
        accept=".pdf,.doc,.docx"
        className="hidden"
        onChange={(e) => {
          const file = e.target.files?.[0];
          e.target.value = "";
          if (file) upload.mutate(file);
        }}
      />
      <Button
        variant="secondary"
        size="sm"
        isLoading={upload.isPending}
        onClick={() => inputRef.current?.click()}
        {...buttonProps}
      >
        <Upload className="size-3.5" aria-hidden />
        {label}
      </Button>
    </>
  );
}
