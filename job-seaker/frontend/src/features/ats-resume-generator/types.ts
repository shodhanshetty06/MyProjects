/**
 * ATS Resume Generator feature — type contracts.
 *
 * Mirrors backend/app/modules/ats_resume_generator/schemas.py.
 */

export type ATSVersionStatus =
  | "pending"
  | "optimizing"
  | "generating_files"
  | "completed"
  | "failed";

export type ATSFileFormat = "docx" | "pdf";

export interface ProjectItem {
  name: string;
  description: string | null;
  technologies: string[];
  url: string | null;
}

export interface ExperienceItem {
  company: string;
  title: string;
  location: string | null;
  start_date: string | null;
  end_date: string | null;
  is_current: boolean;
  description: string | null;
  highlights: string[];
}

export interface EducationItem {
  institution: string;
  degree: string | null;
  field_of_study: string | null;
  start_date: string | null;
  end_date: string | null;
  gpa: string | null;
}

export interface CertificationItem {
  name: string;
  issuer: string | null;
  issue_date: string | null;
  expiry_date: string | null;
  credential_id: string | null;
}

export interface LanguageItem {
  language: string;
  proficiency: string | null;
}

export interface ContactInfo {
  full_name: string;
  phone: string | null;
  email: string | null;
  linkedin_url: string | null;
  github_url: string | null;
  portfolio_url: string | null;
  location: string | null;
}

export interface SkillCategory {
  category: string;
  skills: string[];
}

export interface ATSResumeTemplateSummary {
  id: string;
  name: string;
  description: string;
  body_font: string;
  heading_font: string;
  heading_color: string;
  accent_color: string | null;
}

export interface ATSResumeCreateRequest {
  resume_version_id: string;
  job_description: string;
  title?: string | null;
  job_title?: string | null;
  company_name?: string | null;
  target_job_id?: string | null;
  template_id?: string;
}

export interface ATSResumeVersionCreateRequest {
  job_description?: string | null;
  resume_version_id?: string | null;
  template_id?: string | null;
}

export interface ATSResumeVersionTemplateRequest {
  template_id: string;
}

export interface ATSResumeVersionSummary {
  id: string;
  version_number: number;
  is_current: boolean;
  status: ATSVersionStatus;
  template_id: string;
  keyword_score: number | null;
  ats_score: number | null;
  error_message: string | null;
  validation_warnings: string[];
  available_formats: ATSFileFormat[];
  created_at: string;
}

export interface ATSResumeVersionDetail extends ATSResumeVersionSummary {
  job_description: string;
  source_resume_version_id: string;
  contact: ContactInfo;
  original_summary: string;
  optimized_summary: string;
  original_skills: string[];
  optimized_skills: string[];
  grouped_skills: SkillCategory[];
  original_projects: ProjectItem[];
  optimized_projects: ProjectItem[];
  experience: ExperienceItem[];
  education: EducationItem[];
  certifications: CertificationItem[];
  languages: LanguageItem[];
  achievements: string[];
  keywords: string[];
  matched_keywords: string[];
  missing_keywords: string[];
  score_breakdown: Record<string, unknown>;
  model_used: string | null;
  updated_at: string;
}

export interface ATSResumeRead {
  id: string;
  user_id: string;
  source_resume_id: string;
  target_job_id: string | null;
  title: string;
  job_title: string | null;
  company_name: string | null;
  created_at: string;
  updated_at: string;
  current_version: ATSResumeVersionSummary | null;
}

export interface TextDiffSegment {
  op: "equal" | "insert" | "delete";
  text: string;
}

export interface ListDiff {
  added: string[];
  removed: string[];
  unchanged: string[];
}

export interface ProjectDiff {
  name: string;
  status: "added" | "removed" | "changed" | "unchanged";
  description_diff: TextDiffSegment[];
  highlights_diff: ListDiff;
}

export interface ATSResumeVersionDiff {
  compare_to: string;
  summary_diff: TextDiffSegment[];
  skills_diff: ListDiff;
  projects_diff: ProjectDiff[];
}

export interface ATSResumeDownloadResponse {
  format: ATSFileFormat;
  download_url: string;
}
