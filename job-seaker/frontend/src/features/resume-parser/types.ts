/**
 * Resume Parser feature - type contracts, mirroring
 * backend/app/modules/resume_parser/schemas.py.
 */
export type ResumeVersionStatus = "uploaded" | "parsing" | "parsed" | "failed";
export type ResumeContentType = "pdf" | "docx";

export interface ExperienceItem {
  company: string;
  title: string;
  location?: string | null;
  start_date?: string | null;
  end_date?: string | null;
  is_current: boolean;
  description?: string | null;
  highlights: string[];
}

export interface EducationItem {
  institution: string;
  degree?: string | null;
  field_of_study?: string | null;
  start_date?: string | null;
  end_date?: string | null;
  gpa?: number | null;
}

export interface ProjectItem {
  name: string;
  description?: string | null;
  technologies: string[];
  url?: string | null;
}

export interface CertificationItem {
  name: string;
  issuer?: string | null;
  issue_date?: string | null;
  expiry_date?: string | null;
  credential_id?: string | null;
}

export interface LanguageItem {
  language: string;
  proficiency?: string | null;
}

export interface ParsedResumeData {
  summary: string | null;
  skills: string[];
  experience: ExperienceItem[];
  education: EducationItem[];
  projects: ProjectItem[];
  certifications: CertificationItem[];
  achievements: string[];
  languages: LanguageItem[];
}

export interface ResumeVersionSummary {
  id: string;
  version_number: number;
  is_current: boolean;
  status: ResumeVersionStatus;
  content_type: ResumeContentType;
  original_filename: string;
  file_size_bytes: number;
  error_message: string | null;
  created_at: string;
}

export interface ResumeVersionDetail extends ResumeVersionSummary {
  parsed_data: ParsedResumeData | null;
  has_embedding: boolean;
}

export interface ResumeRead {
  id: string;
  user_id: string;
  title: string;
  created_at: string;
  updated_at: string;
  current_version: ResumeVersionSummary | null;
}

export interface ResumeUpdateRequest {
  title: string;
}
