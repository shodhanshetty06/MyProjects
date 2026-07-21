import { zodResolver } from "@hookform/resolvers/zod";
import { useMutation, useQueryClient } from "@tanstack/react-query";
import { useForm } from "react-hook-form";
import { z } from "zod";

import { Button } from "@/components/ui/Button";
import { Dialog } from "@/components/ui/Dialog";
import { Input } from "@/components/ui/Input";
import { useAuth } from "@/features/auth";
import { createApplication } from "@/features/tracking/api";
import { toast } from "@/stores/toastStore";

const schema = z.object({
  company_name: z.string().min(1, "Company is required"),
  job_title: z.string().min(1, "Job title is required"),
  job_url: z.string().url("Enter a valid URL").optional().or(z.literal("")),
  location: z.string().optional(),
});
type FormValues = z.infer<typeof schema>;

interface CreateApplicationDialogProps {
  open: boolean;
  onOpenChange: (open: boolean) => void;
  onCreated?: (applicationId: string) => void;
}

export function CreateApplicationDialog({ open, onOpenChange, onCreated }: CreateApplicationDialogProps) {
  const { user } = useAuth();
  const userId = user!.id;
  const queryClient = useQueryClient();

  const {
    register,
    handleSubmit,
    reset,
    formState: { errors, isSubmitting },
  } = useForm<FormValues>({ resolver: zodResolver(schema) });

  const mutation = useMutation({
    mutationFn: (values: FormValues) =>
      createApplication(userId, {
        company_name: values.company_name,
        job_title: values.job_title,
        job_url: values.job_url || null,
        location: values.location || null,
        source: "manual",
      }),
    onSuccess: (app) => {
      toast.success("Application added");
      void queryClient.invalidateQueries({ queryKey: ["applications", userId] });
      void queryClient.invalidateQueries({ queryKey: ["tracking-dashboard", userId] });
      reset();
      onOpenChange(false);
      onCreated?.(app.id);
    },
    onError: () => toast.danger("Couldn't add this application"),
  });

  return (
    <Dialog
      open={open}
      onOpenChange={onOpenChange}
      title="Track a new application"
      footer={
        <>
          <Button variant="ghost" onClick={() => onOpenChange(false)}>
            Cancel
          </Button>
          <Button onClick={handleSubmit((v) => mutation.mutate(v))} isLoading={isSubmitting || mutation.isPending}>
            Add application
          </Button>
        </>
      }
    >
      <form className="space-y-3" onSubmit={(e) => e.preventDefault()}>
        <div className="space-y-1.5">
          <label className="text-xs font-medium text-ink-secondary">Company</label>
          <Input invalid={!!errors.company_name} {...register("company_name")} />
          {errors.company_name && <p className="text-xs text-danger">{errors.company_name.message}</p>}
        </div>
        <div className="space-y-1.5">
          <label className="text-xs font-medium text-ink-secondary">Job title</label>
          <Input invalid={!!errors.job_title} {...register("job_title")} />
          {errors.job_title && <p className="text-xs text-danger">{errors.job_title.message}</p>}
        </div>
        <div className="space-y-1.5">
          <label className="text-xs font-medium text-ink-secondary">Job URL (optional)</label>
          <Input invalid={!!errors.job_url} {...register("job_url")} />
          {errors.job_url && <p className="text-xs text-danger">{errors.job_url.message}</p>}
        </div>
        <div className="space-y-1.5">
          <label className="text-xs font-medium text-ink-secondary">Location (optional)</label>
          <Input {...register("location")} />
        </div>
      </form>
    </Dialog>
  );
}
