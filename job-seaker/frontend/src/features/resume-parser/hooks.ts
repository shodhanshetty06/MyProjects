import { useEffect, useState } from "react";
import { useQuery } from "@tanstack/react-query";

import { listResumes } from "@/features/resume-parser/api";
import type { ResumeVersionStatus } from "@/features/resume-parser/types";

/** Parsed resume versions as `{ value, label }` options, for pickers across other features. */
export function useParsedResumeOptions(userId: string) {
  const query = useQuery({
    queryKey: ["resumes", userId, "picker"],
    queryFn: () => listResumes(userId, 1, 50),
  });

  const options = (query.data?.data ?? [])
    .filter((r) => r.current_version?.status === "parsed")
    .map((r) => ({ value: r.current_version!.id, label: r.title }));

  return { options, isLoading: query.isLoading };
}

const PARSING_STAGES = [
  "Extracting text from your file...",
  "Reading your work experience...",
  "Identifying your skills...",
  "Structuring your education and projects...",
  "Polishing the final details...",
];

const STAGE_INTERVAL_MS = 2500;

/**
 * The backend only reports coarse "uploaded" | "parsing" status for this step
 * (the AI extraction call is a single request with no sub-progress to report),
 * so this cycles through plausible stage labels client-side to keep the wait
 * feeling active instead of a single static spinner.
 */
export function useParsingStageMessage(status: ResumeVersionStatus | undefined): string {
  const [stageIndex, setStageIndex] = useState(0);

  useEffect(() => {
    if (status !== "parsing") {
      setStageIndex(0);
      return;
    }
    const interval = setInterval(() => {
      setStageIndex((i) => Math.min(i + 1, PARSING_STAGES.length - 1));
    }, STAGE_INTERVAL_MS);
    return () => clearInterval(interval);
  }, [status]);

  if (status === "uploaded") return "Uploading your file...";
  return PARSING_STAGES[stageIndex];
}
