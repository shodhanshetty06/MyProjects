import { Eye, EyeOff } from "lucide-react";
import { forwardRef, useState } from "react";

import { Input, type InputProps } from "@/components/ui/Input";

export const PasswordInput = forwardRef<HTMLInputElement, Omit<InputProps, "type" | "trailingIcon">>(
  (props, ref) => {
    const [visible, setVisible] = useState(false);

    return (
      <Input
        ref={ref}
        type={visible ? "text" : "password"}
        trailingIcon={
          <button
            type="button"
            tabIndex={-1}
            onClick={() => setVisible((current) => !current)}
            aria-label={visible ? "Hide password" : "Show password"}
            className="pointer-events-auto flex text-ink-muted hover:text-ink-secondary"
          >
            {visible ? <EyeOff className="size-4" aria-hidden /> : <Eye className="size-4" aria-hidden />}
          </button>
        }
        {...props}
      />
    );
  },
);
PasswordInput.displayName = "PasswordInput";
