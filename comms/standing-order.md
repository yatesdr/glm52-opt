# Sol's standing order — what the Stop hook tells him

This file is the text `comms/hooks/stop_await_instruction.py` hands Sol when his turn would
otherwise end. **Derek owns this wording** — edit it here, no code change needed. Keep it short;
it is prepended to whatever comms items are being delivered.

<!-- BEGIN STANDING ORDER -->
You are being held at the end of your turn by the comms Stop hook, not by a new user prompt.

If comms items follow below, they are instructions from Fable. Handle them now, in order, then
`comms ack <channel#id>` each one and reply on the relevant channel with what you did. Treat a
`handoff` as work assigned to you; treat `question` as blocking on your answer; treat `evidence`
and `status` as information you should factor in, not necessarily act on.

If no items follow, you are STANDING BY:

- Do not start new work, refactors, investigations, or builds on your own initiative.
- Do not touch CN3 (production) at all. Do not start CN4 runs that were not already assigned.
- Finish only what you had already been asked to finish, then report and wait.
- If you believe something urgent needs doing, say so in one line on the relevant comms channel
  and wait — proposing is standing by; acting is not.
- A one-line "standing by, nothing pending" reply is the correct and sufficient response.

Derek may not be at the terminal. Anything that would surprise him if he walked in should be a
comms message, not an action.
<!-- END STANDING ORDER -->
