import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { motion } from "framer-motion";
import { Bot, Plus, Send, Trash2, User } from "lucide-react";
import { useEffect, useRef, useState } from "react";

import { PageHeader } from "@/components/layout/PageHeader";
import { ResizableSplit } from "@/components/layout/ResizableSplit";
import { staggerChildren } from "@/components/motion/variants";
import { Button } from "@/components/ui/Button";
import { EmptyState } from "@/components/ui/EmptyState";
import { SkeletonText } from "@/components/ui/Skeleton";
import { useAuth } from "@/features/auth";
import { deleteConversation, getConversation, listConversations, sendChatMessage } from "@/features/ai-career-coach/api";
import { ThinkingIndicator } from "@/features/ai-career-coach/components/ThinkingIndicator";
import { Typewriter } from "@/features/ai-career-coach/components/Typewriter";
import type { CoachMessageRead } from "@/features/ai-career-coach/types";
import { cn } from "@/lib/cn";
import { formatRelativeDate } from "@/lib/format";

export function CoachChatPage() {
  const { user } = useAuth();
  const userId = user!.id;
  const queryClient = useQueryClient();
  const [conversationId, setConversationId] = useState<string | null>(null);
  const [input, setInput] = useState("");
  const [pendingUserMessage, setPendingUserMessage] = useState<string | null>(null);
  const [animatingMessageId, setAnimatingMessageId] = useState<string | null>(null);
  const scrollRef = useRef<HTMLDivElement>(null);

  const conversationsQuery = useQuery({
    queryKey: ["coach-conversations", userId],
    queryFn: () => listConversations(userId, 1, 30),
  });

  const activeConversationQuery = useQuery({
    queryKey: ["coach-conversation", userId, conversationId],
    queryFn: () => getConversation(userId, conversationId as string),
    enabled: !!conversationId,
  });

  const sendMutation = useMutation({
    mutationFn: (message: string) => sendChatMessage(userId, message, conversationId),
    onMutate: (message) => setPendingUserMessage(message),
    onSuccess: async (response) => {
      setConversationId(response.conversation.id);
      setPendingUserMessage(null);
      setInput("");
      await queryClient.invalidateQueries({ queryKey: ["coach-conversation", userId, response.conversation.id] });
      await queryClient.invalidateQueries({ queryKey: ["coach-conversations", userId] });
      setAnimatingMessageId(response.message.id);
    },
    onError: () => setPendingUserMessage(null),
  });

  const deleteMutation = useMutation({
    mutationFn: (id: string) => deleteConversation(userId, id),
    onSuccess: () => {
      if (conversationId) setConversationId(null);
      void queryClient.invalidateQueries({ queryKey: ["coach-conversations", userId] });
    },
  });

  useEffect(() => {
    scrollRef.current?.scrollTo({ top: scrollRef.current.scrollHeight, behavior: "smooth" });
  }, [activeConversationQuery.data, pendingUserMessage]);

  const messages: CoachMessageRead[] = activeConversationQuery.data?.messages ?? [];

  function handleSend() {
    const text = input.trim();
    if (!text || sendMutation.isPending) return;
    sendMutation.mutate(text);
  }

  return (
    <div>
      <PageHeader eyebrow="AI Career Coach" title="Chat" description="Ask anything about your resume, search, or strategy." />

      <ResizableSplit
        persistKey="coach-chat"
        leftDefaultSize={72}
        left={
          <div className="flex h-[calc(100vh-220px)] min-h-[420px] flex-col rounded-[var(--radius-lg)] border border-border bg-surface pr-3">
            <div ref={scrollRef} className="flex-1 space-y-4 overflow-y-auto p-4">
              {messages.length === 0 && !pendingUserMessage ? (
                <EmptyState icon={Bot} title="Start a conversation" description="Ask about your resume, a target role, or how to close a skill gap." />
              ) : (
                <motion.div variants={staggerChildren(0.04)} initial="hidden" animate="visible" className="space-y-4">
                  {messages.map((message) => (
                    <ChatBubble
                      key={message.id}
                      message={message}
                      animate={message.id === animatingMessageId}
                      onAnimateDone={() => setAnimatingMessageId(null)}
                    />
                  ))}
                  {pendingUserMessage && (
                    <ChatBubble message={{ id: "pending", role: "user", content: pendingUserMessage, created_at: new Date().toISOString() }} />
                  )}
                  {sendMutation.isPending && (
                    <div className="flex items-start gap-2.5">
                      <div className="flex size-7 shrink-0 items-center justify-center rounded-full bg-brand-soft text-brand">
                        <Bot className="size-3.5" aria-hidden />
                      </div>
                      <ThinkingIndicator />
                    </div>
                  )}
                </motion.div>
              )}
            </div>
            <div className="flex items-center gap-2 border-t border-border p-3">
              <input
                value={input}
                onChange={(e) => setInput(e.target.value)}
                onKeyDown={(e) => {
                  if (e.key === "Enter" && !e.shiftKey) {
                    e.preventDefault();
                    handleSend();
                  }
                }}
                placeholder="Message your career coach..."
                className="h-10 flex-1 rounded-[var(--radius-md)] border border-border bg-bg px-3 text-sm text-ink outline-none focus:border-brand focus:ring-4 focus:ring-brand-soft"
              />
              <Button onClick={handleSend} disabled={!input.trim() || sendMutation.isPending} isLoading={sendMutation.isPending}>
                <Send className="size-4" aria-hidden />
              </Button>
            </div>
          </div>
        }
        right={
          <div className="pl-3">
            <div className="mb-2 flex items-center justify-between">
              <p className="text-xs font-semibold uppercase tracking-wide text-ink-muted">History</p>
              <Button size="sm" variant="ghost" onClick={() => setConversationId(null)}>
                <Plus className="size-3.5" aria-hidden />
                New
              </Button>
            </div>
            {conversationsQuery.isLoading ? (
              <SkeletonText lines={4} />
            ) : (
              <div className="space-y-1">
                {(conversationsQuery.data?.data ?? []).map((conv) => (
                  <button
                    key={conv.id}
                    onClick={() => setConversationId(conv.id)}
                    className={cn(
                      "group flex w-full items-center gap-2 rounded-[var(--radius-md)] px-2.5 py-2 text-left text-xs transition-colors",
                      conv.id === conversationId ? "bg-brand-soft text-brand" : "text-ink-secondary hover:bg-surface-2",
                    )}
                  >
                    <span className="min-w-0 flex-1 truncate">{conv.title || "New conversation"}</span>
                    <span className="shrink-0 text-[10px] text-ink-muted">{formatRelativeDate(conv.last_message_at)}</span>
                    <Trash2
                      className="size-3 shrink-0 opacity-0 transition-opacity hover:text-danger group-hover:opacity-100"
                      onClick={(e) => {
                        e.stopPropagation();
                        deleteMutation.mutate(conv.id);
                      }}
                    />
                  </button>
                ))}
              </div>
            )}
          </div>
        }
      />
    </div>
  );
}

function ChatBubble({
  message,
  animate,
  onAnimateDone,
}: {
  message: CoachMessageRead;
  animate?: boolean;
  onAnimateDone?: () => void;
}) {
  const isUser = message.role === "user";
  return (
    <motion.div
      variants={{ hidden: { opacity: 0, y: 6 }, visible: { opacity: 1, y: 0 } }}
      className={cn("flex items-start gap-2.5", isUser && "flex-row-reverse")}
    >
      <div
        className={cn(
          "flex size-7 shrink-0 items-center justify-center rounded-full",
          isUser ? "bg-surface-2 text-ink-muted" : "bg-brand-soft text-brand",
        )}
      >
        {isUser ? <User className="size-3.5" aria-hidden /> : <Bot className="size-3.5" aria-hidden />}
      </div>
      <div
        className={cn(
          "max-w-[75%] whitespace-pre-wrap rounded-[var(--radius-lg)] px-3.5 py-2.5 text-sm leading-relaxed",
          isUser ? "bg-brand text-brand-ink" : "bg-surface-2 text-ink",
        )}
      >
        {animate ? <Typewriter text={message.content} onDone={onAnimateDone} /> : message.content}
      </div>
    </motion.div>
  );
}
