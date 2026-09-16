# Decision Log

1. **Brand selection: GWRHelp, not a top-20-by-volume brand.** Picked using conversation-level
   engagement stats (64.77% multi-turn, 43.78% with 2+ company responses) rather than raw tweet
   volume, on the hypothesis that a brand which actually engages in-thread gives better grounding
   data than a high-volume brand that mostly deflects to DMs.

2. **Thread reconstruction over raw rows.** The dataset has no `conversation_id` — threads were
   rebuilt from `tweet_id` / `in_response_to_tweet_id` / `response_tweet_id`, sorted by
   `created_at` since row order isn't chronological. Had to perform this before even starting simple EDA to data prep.

3. **[Branching resolution strategy]** — `response_tweet_id` can contain multiple ids, making
   threads trees rather than lines. [State how you flattened branches — e.g. kept the longest
   customer↔brand alternating path, dropped other-user side replies.]

4. **Zero-leakage quarantine.** The golden evaluation set was carved out of the conversation pool
   *before* building the retrieval index or few-shot pool, so no golden example can be retrieved
   as its own grounding evidence during evaluation. This is kinda obvious but adding this here to log.

5. **[Intent taxonomy size: 12 classes]** — [State how you arrived at 12 specifically — e.g.
   embedding + clustering pass, then manual merge of overlapping clusters — and why you stopped
   there rather than going more granular or coarser.] - Actually this 12 class have a overall after performing judging, now since i cant reconstruct from start, but if i have one more week i would try to eliminate overlaps and add more distinctive classes.

6. **4-way decision routing instead of binary auto-handle/escalate.** Added `ASK_CLARIFICATION`
   and `REQUEST_INFORMATION` as distinct outcomes from plain `RESPOND`/`ESCALATE`, since a
   meaningful share of real GWR threads needed more info from the customer before either was
   possible — collapsing those into a binary would have forced a less accurate label. Most of the multi-thread conv were like people stating about something but gives more abstract details and support responds with a clarifying questions so implemented this.

7. **No UI.** Output is CSV + terminal rather than a web interface — faster to audit line-by-line
   and avoids building a UI layer the assignment doesn't ask for. Also showing the back actions in UI is kinda hard but terminal prints + CSV
   give much more details and better insights.

8. **[Embedding model: all-MiniLM-L6-v2]** — i see most of the conversations are 2-5 turns for basic query which AI can handle, multiple rounds or often escalated based on my observation so, planned to use a simpler model for embedding for speed up the retrieval process. and appropriate embedding dimension to work with. 

9. **[FAISS vector search index]** — since its a prototype, implementing a simple search index is enough to meet the requirement. and it is efficient for single user and small scale, no need to complicate it with vector DB management.

10. **[Sliding-window chunking for long threads]** — while drafting this decisions i had an idea when a particular embedding part comes to context take that entire surrounding conversation seem to make more sense can implement in future, right now this is to retain information which is inside window size for context of next few words.

11. **[Baseline definitions]** — since i had to do this project with very short time, i had to take as a constant message as a baseline, but i had a better idea to use TF-IDF + cosine similarity score to retrieve relevant context, but i could only do keyword as baseline2, future we can add this one as good baseline at some case this can perform better than the vector embedding retrievals. 

13. **[Resolved-thread filtering heuristic]** — see this is important because the escalation step basically works based on this where this similar issue have been resolved or asked to DM which gives when a issue needs to be escalated or not.
