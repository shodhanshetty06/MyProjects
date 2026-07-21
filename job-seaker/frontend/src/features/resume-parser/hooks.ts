import { useQuery } from "@tanstack/react-query";

import { listResumes } from "@/features/resume-parser/api";

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
