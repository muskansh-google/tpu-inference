import sys

with open('persistent_batch_manager.py', 'r') as f:
    lines = f.readlines()

out = []
in_fn = False
for i, line in enumerate(lines):
    if "def update_states(self" in line:
        in_fn = True
        out.append(line)
        continue
    
    if in_fn:
        if line.strip() == 'return batch_changed':
            out.append("        return batch_changed\n")
            out.append("    except Exception as e:\n")
            out.append("        import traceback\n")
            out.append("        print(f\"DEBUG_PBM: Exception caught in update_states: {e}\", flush=True)\n")
            out.append("        traceback.print_exc()\n")
            out.append("        raise\n")
            in_fn = False
        elif line.strip() == '"""Update the cached states and the persistent batch with the scheduler':
            out.append(line)
            # Find end of docstring
            for j in range(i+1, len(lines)):
                out.append(lines[j])
                if lines[j].strip() == '"""':
                    # Document string ended, insert try block and prints here
                    out.append("    try:\n")
                    out.append("        print(f\"DEBUG_PBM: update_states called.\", flush=True)\n")
                    out.append("        if hasattr(scheduler_output, 'scheduled_new_reqs'):\n")
                    out.append("            print(f\"DEBUG_PBM: scheduled_new_reqs={[r.req_id for r in scheduler_output.scheduled_new_reqs]}\", flush=True)\n")
                    out.append("        if hasattr(scheduler_output, 'scheduled_cached_reqs'):\n")
                    out.append("            print(f\"DEBUG_PBM: scheduled_cached_reqs={scheduler_output.scheduled_cached_reqs.req_ids}\", flush=True)\n")
                    out.append("        print(f\"DEBUG_PBM: num_scheduled_tokens={scheduler_output.num_scheduled_tokens}\", flush=True)\n")
                    out.append("        print(f\"DEBUG_PBM: scheduled_req_ids={scheduler_output.num_scheduled_tokens.keys()}\", flush=True)\n")
                    break
        elif getattr(line, "strip", lambda: "")() in [
            'output.', 
            'The updated states are used by the `_prepare_inputs` function to create',
            'the input TPU tensors for the model.',
            'Returns:',
            'True if there is a new/resumed/paused/finished request.',
            'If False, we can skip copying SamplingMetadata to the TPU.',
            '"""'
        ]:
            pass # Already appended inside docstring loop
        else:
            out.append("    " + line)
    else:
        out.append(line)

with open('persistent_batch_manager.py', 'w') as f:
    f.writelines(out)
