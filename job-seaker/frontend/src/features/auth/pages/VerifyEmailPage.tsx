import { MailCheck } from "lucide-react";
import { Link } from "react-router-dom";

import { Button } from "@/components/ui/Button";
import { NotConnectedNotice } from "@/components/ui/NotConnectedNotice";
import { useAuth } from "@/features/auth/AuthProvider";
import { paths } from "@/routes/paths";

export function VerifyEmailPage() {
  const { user } = useAuth();

  return (
    <div className="text-center">
      <div className="mx-auto flex size-11 items-center justify-center rounded-full bg-brand-soft text-brand">
        <MailCheck className="size-5" aria-hidden />
      </div>
      <h1 className="mt-4 text-lg font-semibold text-ink">Verify your email</h1>
      <p className="mt-1 text-sm text-ink-secondary">
        {user ? `We'd send a confirmation link to ${user.email}.` : "We'd send a confirmation link to your inbox."}
      </p>

      <NotConnectedNotice
        className="mt-6 text-left"
        description="Email verification isn't implemented on this backend yet - your account works normally without it."
      />

      <Link to={user ? paths.workspace : paths.login} className="mt-6 block">
        <Button className="w-full">{user ? "Continue to app" : "Back to log in"}</Button>
      </Link>
    </div>
  );
}
