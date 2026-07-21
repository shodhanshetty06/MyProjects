import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { useEffect, useState } from "react";

import { Badge } from "@/components/ui/Badge";
import { Button } from "@/components/ui/Button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/Card";
import { Checkbox } from "@/components/ui/Checkbox";
import { Input } from "@/components/ui/Input";
import { SkeletonText } from "@/components/ui/Skeleton";
import { useAuth } from "@/features/auth";
import { getApplicantProfile, upsertApplicantProfile } from "@/features/auto-apply/api";
import type { ApplicantProfileUpsertRequest } from "@/features/auto-apply/types";
import { SettingsShell } from "@/features/settings/components/SettingsShell";
import { toast } from "@/stores/toastStore";

const EMPTY: ApplicantProfileUpsertRequest = {
  auto_apply_enabled: false,
  first_name: null,
  last_name: null,
  email: null,
  phone: null,
  address_line1: null,
  city: null,
  state: null,
  postal_code: null,
  country: null,
  linkedin_url: null,
  portfolio_url: null,
  github_url: null,
  website_url: null,
  work_authorized: null,
  requires_sponsorship: null,
  willing_to_relocate: null,
  years_of_experience: null,
  desired_salary: null,
  notice_period: null,
  gender: null,
  race_ethnicity: null,
  veteran_status: null,
  disability_status: null,
  default_ats_resume_id: null,
  default_resume_id: null,
  custom_answers: {},
};

function field(value: string | null): string {
  return value ?? "";
}

export function SettingsProfilePage() {
  const { user } = useAuth();
  const userId = user!.id;
  const queryClient = useQueryClient();
  const [form, setForm] = useState<ApplicantProfileUpsertRequest>(EMPTY);

  const profileQuery = useQuery({
    queryKey: ["applicant-profile", userId],
    queryFn: () => getApplicantProfile(userId),
  });

  useEffect(() => {
    const profile = profileQuery.data;
    if (!profile) return;
    setForm({
      auto_apply_enabled: profile.auto_apply_enabled,
      first_name: profile.first_name,
      last_name: profile.last_name,
      email: profile.email,
      phone: profile.phone,
      address_line1: profile.address_line1,
      city: profile.city,
      state: profile.state,
      postal_code: profile.postal_code,
      country: profile.country,
      linkedin_url: profile.linkedin_url,
      portfolio_url: profile.portfolio_url,
      github_url: profile.github_url,
      website_url: profile.website_url,
      work_authorized: profile.work_authorized,
      requires_sponsorship: profile.requires_sponsorship,
      willing_to_relocate: profile.willing_to_relocate,
      years_of_experience: profile.years_of_experience,
      desired_salary: profile.desired_salary,
      notice_period: profile.notice_period,
      gender: profile.gender,
      race_ethnicity: profile.race_ethnicity,
      veteran_status: profile.veteran_status,
      disability_status: profile.disability_status,
      default_ats_resume_id: profile.default_ats_resume_id,
      default_resume_id: profile.default_resume_id,
      custom_answers: profile.custom_answers,
    });
  }, [profileQuery.data]);

  const saveMutation = useMutation({
    mutationFn: () => upsertApplicantProfile(userId, form),
    onSuccess: () => {
      toast.success("Profile saved");
      void queryClient.invalidateQueries({ queryKey: ["applicant-profile", userId] });
    },
    onError: () => toast.danger("Couldn't save profile"),
  });

  function set<K extends keyof ApplicantProfileUpsertRequest>(key: K, value: ApplicantProfileUpsertRequest[K]) {
    setForm((prev) => ({ ...prev, [key]: value }));
  }

  return (
    <SettingsShell>
      <div className="space-y-6">
        <Card>
          <CardHeader>
            <CardTitle>Account</CardTitle>
          </CardHeader>
          <CardContent className="flex flex-wrap items-center gap-3">
            <span className="text-sm text-ink">{user?.email}</span>
            <Badge tone={user?.is_active ? "success" : "neutral"}>{user?.is_active ? "Active" : "Inactive"}</Badge>
            {user?.created_at && <span className="text-xs text-ink-muted">Member since {new Date(user.created_at).toLocaleDateString()}</span>}
          </CardContent>
        </Card>

        {profileQuery.isLoading ? (
          <SkeletonText lines={8} />
        ) : (
          <>
            <Card>
              <CardHeader>
                <CardTitle>Applicant profile</CardTitle>
              </CardHeader>
              <CardContent className="space-y-4">
                <p className="text-xs text-ink-muted">
                  Used to pre-fill auto-apply forms. This data lives with the auto-apply module, not a separate profile
                  service.
                </p>
                <label className="flex items-center gap-2 text-sm text-ink">
                  <Checkbox checked={!!form.auto_apply_enabled} onCheckedChange={(v) => set("auto_apply_enabled", v)} />
                  Enable auto apply
                </label>

                <div className="grid grid-cols-2 gap-3">
                  <Field label="First name" value={form.first_name} onChange={(v) => set("first_name", v)} />
                  <Field label="Last name" value={form.last_name} onChange={(v) => set("last_name", v)} />
                  <Field label="Email" value={form.email} onChange={(v) => set("email", v)} />
                  <Field label="Phone" value={form.phone} onChange={(v) => set("phone", v)} />
                </div>
              </CardContent>
            </Card>

            <Card>
              <CardHeader>
                <CardTitle>Address</CardTitle>
              </CardHeader>
              <CardContent className="grid grid-cols-2 gap-3">
                <Field label="Address line" value={form.address_line1} onChange={(v) => set("address_line1", v)} className="col-span-2" />
                <Field label="City" value={form.city} onChange={(v) => set("city", v)} />
                <Field label="State / Region" value={form.state} onChange={(v) => set("state", v)} />
                <Field label="Postal code" value={form.postal_code} onChange={(v) => set("postal_code", v)} />
                <Field label="Country" value={form.country} onChange={(v) => set("country", v)} />
              </CardContent>
            </Card>

            <Card>
              <CardHeader>
                <CardTitle>Links</CardTitle>
              </CardHeader>
              <CardContent className="grid grid-cols-2 gap-3">
                <Field label="LinkedIn" value={form.linkedin_url} onChange={(v) => set("linkedin_url", v)} />
                <Field label="Portfolio" value={form.portfolio_url} onChange={(v) => set("portfolio_url", v)} />
                <Field label="GitHub" value={form.github_url} onChange={(v) => set("github_url", v)} />
                <Field label="Website" value={form.website_url} onChange={(v) => set("website_url", v)} />
              </CardContent>
            </Card>

            <Card>
              <CardHeader>
                <CardTitle>Work preferences</CardTitle>
              </CardHeader>
              <CardContent className="space-y-3">
                <div className="flex flex-wrap gap-4">
                  <label className="flex items-center gap-2 text-sm text-ink">
                    <Checkbox checked={!!form.work_authorized} onCheckedChange={(v) => set("work_authorized", v)} />
                    Work authorized
                  </label>
                  <label className="flex items-center gap-2 text-sm text-ink">
                    <Checkbox checked={!!form.requires_sponsorship} onCheckedChange={(v) => set("requires_sponsorship", v)} />
                    Requires sponsorship
                  </label>
                  <label className="flex items-center gap-2 text-sm text-ink">
                    <Checkbox checked={!!form.willing_to_relocate} onCheckedChange={(v) => set("willing_to_relocate", v)} />
                    Willing to relocate
                  </label>
                </div>
                <div className="grid grid-cols-3 gap-3">
                  <Field
                    label="Years of experience"
                    type="number"
                    value={form.years_of_experience?.toString() ?? ""}
                    onChange={(v) => set("years_of_experience", v ? Number(v) : null)}
                  />
                  <Field
                    label="Desired salary"
                    type="number"
                    value={form.desired_salary?.toString() ?? ""}
                    onChange={(v) => set("desired_salary", v ? Number(v) : null)}
                  />
                  <Field label="Notice period" value={form.notice_period} onChange={(v) => set("notice_period", v)} />
                </div>
              </CardContent>
            </Card>

            <div className="flex justify-end">
              <Button onClick={() => saveMutation.mutate()} isLoading={saveMutation.isPending}>
                Save profile
              </Button>
            </div>
          </>
        )}
      </div>
    </SettingsShell>
  );
}

function Field({
  label,
  value,
  onChange,
  type = "text",
  className,
}: {
  label: string;
  value: string | null | undefined;
  onChange: (value: string) => void;
  type?: string;
  className?: string;
}) {
  return (
    <div className={`space-y-1.5 ${className ?? ""}`}>
      <label className="text-xs font-medium text-ink-secondary">{label}</label>
      <Input type={type} value={field(value ?? null)} onChange={(e) => onChange(e.target.value)} />
    </div>
  );
}
