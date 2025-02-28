
def display_top_contexts_for_feature(
   model, # only needed for tokenizer
   tokens,
   top_acts,
   top_inds,
   feature_id,
   num_to_show=10,
   left_context=20,
   right_context=5,
   min_opacity=0.2,
   minimal=False
):
   """
   Display the top activating contexts for a feature using HTML with translucent highlighting.
   Only shows one example per unique context.
   
   Args:
       model: HuggingFace model with tokenizer
       tokens: tensor of shape [batch, n_ctx] containing token ids
       top_acts: tensor of shape [batch, n_ctx, k] containing activation values
       top_inds: tensor of shape [batch, n_ctx, k] containing feature indices
       feature_id: which feature to analyze
       num_to_show: number of top contexts to display
       left_context: number of tokens to show before the max activation
       right_context: number of tokens to show after the max activation
       min_opacity: minimum opacity for highlighting
       minimal: if True, removes labels and spacing between contexts
   """
   import torch
   from IPython.display import HTML
   import html
   
   def get_color(act_value, max_act):
       """Generate translucent red color based on activation value."""
       if act_value == 0:
           return "rgba(0, 0, 0, 0)"
       opacity = min(1.0, max(min_opacity, act_value / max_act))
       return f"rgba(255, 0, 0, {opacity})"
   
   html_output = """
   <div style="font-family: monospace; background-color: #1a1a1a; padding: 20px;">
       <style>
           .token {
               display: inline;
               padding: 0;
               margin: 0;
               position: relative;
           }
           .token:hover .tooltip {
               display: block;
           }
           .tooltip {
               display: none;
               position: absolute;
               background: #333;
               color: white;
               padding: 0px 0px;
               border-radius: 0px;
               font-size: 12px;
               bottom: 100%;
               left: 50%;
               transform: translateX(-50%);
               white-space: nowrap;
               z-index: 1;
           }
           .context-box {
               padding: 10px;
               border: 1px solid #333;
               border-radius: 4px;
               background: #252525;
           }
           .context-box-minimal {
               padding: 3px;
               background: #252525;
           }
           .text-container {
               font-size: 0;
               word-spacing: 0;
               letter-spacing: 0;
           }
           .activation-label {
               color: #888;
               margin-bottom: 8px;
           }
       </style>
   """
   
   # Find positions where this feature appears in top_inds
   batch_idxs, pos_idxs, k_idxs = torch.where(top_inds == feature_id)
   
   # Get the activation values for this feature
   acts = top_acts[batch_idxs, pos_idxs, k_idxs]
   
   # Sort by activation value
   sorted_idxs = torch.argsort(acts, descending=True)
   
   # Keep track of seen contexts to ensure uniqueness
   seen_contexts = set()
   unique_contexts = []
   
   # Get unique contexts
   for idx in sorted_idxs:
       batch_idx = batch_idxs[idx]
       tok_pos = pos_idxs[idx]
       activation = acts[idx].item()
       
       # Get context window for comparison
       start_idx = max(0, tok_pos - left_context)
       end_idx = min(tokens.shape[1], tok_pos + right_context + 1)
       context_tokens = tokens[batch_idx, start_idx:end_idx].tolist()
       
       # Create a hashable representation of the context
       context_key = tuple(context_tokens)
       
       if context_key not in seen_contexts:
           seen_contexts.add(context_key)
           unique_contexts.append((batch_idx, tok_pos, activation))
           
           if len(unique_contexts) == num_to_show:
               break
   
   # Display each unique context
   for batch_idx, tok_pos, activation in unique_contexts:
       # Get context window
       start_idx = max(0, tok_pos - left_context)
       end_idx = min(tokens.shape[1], tok_pos + right_context + 1)
       context_tokens = tokens[batch_idx, start_idx:end_idx]
       
       box_class = "context-box-minimal" if minimal else "context-box"
       margin_style = "" if minimal else "margin: 20px 0;"
       
       html_output += f"""
       <div class="{box_class}" style="{margin_style}">"""
       
       if not minimal:
           html_output += f"""
           <div class="activation-label">activation: {activation:.4f}</div>"""
           
       html_output += """
           <div class="text-container">
       """
       
       # Process each token in the context
       for j, token in enumerate(context_tokens):
           text = model.tokenizer.decode([token])
           pos = start_idx + j
           
           if pos == tok_pos:
               # Max activation token
               color = get_color(activation, activation)
           elif pos in pos_idxs[batch_idxs == batch_idx]:
               # Other activations of this feature
               local_act_idx = torch.where((batch_idxs == batch_idx) & (pos_idxs == pos))[0][0]
               local_act = acts[local_act_idx].item()
               color = get_color(local_act, activation)
           else:
               # Regular context
               color = "rgba(0, 0, 0, 0)"
           
           tooltip_text = f"Activation: {activation:.4f}" if pos == tok_pos else ""
           
           html_output += f"""
               <span class="token" style="background-color: {color}; color: #fff; font-size: 16px;">
                   {html.escape(text)}
                   <span class="tooltip">{tooltip_text}</span>
               </span>"""
       
       html_output += """
           </div>
       </div>
       """
   
   html_output += "</div>"
   return HTML(html_output)