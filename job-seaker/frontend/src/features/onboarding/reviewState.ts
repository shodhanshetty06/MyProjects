import type {
  CertificationItem,
  ContactInfo,
  EducationItem,
  ExperienceItem,
  LanguageItem,
  ParsedResumeData,
  ProjectItem,
} from "@/features/resume-parser/types";

export type ItemSource = "ai" | "added";

export interface ReviewItem<T> {
  id: string;
  source: ItemSource;
  accepted: boolean;
  data: T;
}

let idCounter = 0;
export function nextReviewId(): string {
  idCounter += 1;
  return `review-item-${idCounter}`;
}

function toReviewItems<T>(items: T[]): ReviewItem<T>[] {
  return items.map((data) => ({ id: nextReviewId(), source: "ai", accepted: false, data }));
}

export interface ProfileReviewState {
  contact: ContactInfo;
  summary: string;
  targetRoles: string[];
  workAuthorization: string;
  skills: string[];
  achievements: string[];
  experience: ReviewItem<ExperienceItem>[];
  education: ReviewItem<EducationItem>[];
  projects: ReviewItem<ProjectItem>[];
  certifications: ReviewItem<CertificationItem>[];
  languages: ReviewItem<LanguageItem>[];
}

export const EMPTY_EXPERIENCE: ExperienceItem = {
  company: "",
  title: "",
  location: null,
  start_date: null,
  end_date: null,
  is_current: false,
  description: null,
  highlights: [],
};

export const EMPTY_EDUCATION: EducationItem = {
  institution: "",
  degree: null,
  field_of_study: null,
  start_date: null,
  end_date: null,
  gpa: null,
  gpa_type: null,
};

export const EMPTY_PROJECT: ProjectItem = { name: "", description: null, technologies: [], url: null };

export const EMPTY_CERTIFICATION: CertificationItem = {
  name: "",
  issuer: null,
  issue_date: null,
  expiry_date: null,
  credential_id: null,
};

export const EMPTY_LANGUAGE: LanguageItem = { language: "", proficiency: null };

/**
 * Seeds the Profile Review screen from the AI extraction. `target_roles`
 * has no direct resume field, so it's suggested from the (deduped) titles
 * of the parsed work history — still AI-derived, just not a literal
 * passthrough — and `work_authorization` has no resume signal at all, so it
 * starts genuinely empty ("Missing") for the user to fill in.
 */
export function buildInitialReviewState(parsed: ParsedResumeData): ProfileReviewState {
  const suggestedRoles = Array.from(
    new Set(
      parsed.experience
        .map((exp) => exp.title?.trim())
        .filter((title): title is string => !!title),
    ),
  ).slice(0, 3);

  return {
    contact: parsed.contact,
    summary: parsed.summary ?? "",
    targetRoles: suggestedRoles,
    workAuthorization: "",
    skills: parsed.skills,
    achievements: parsed.achievements,
    experience: toReviewItems(parsed.experience),
    education: toReviewItems(parsed.education),
    projects: toReviewItems(parsed.projects),
    certifications: toReviewItems(parsed.certifications),
    languages: toReviewItems(parsed.languages),
  };
}

export function toParsedResumeData(state: ProfileReviewState): ParsedResumeData {
  return {
    contact: state.contact,
    summary: state.summary,
    skills: state.skills,
    achievements: state.achievements,
    experience: state.experience.map((item) => item.data),
    education: state.education.map((item) => item.data),
    projects: state.projects.map((item) => item.data),
    certifications: state.certifications.map((item) => item.data),
    languages: state.languages.map((item) => item.data),
  };
}
