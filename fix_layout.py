import re
import os

filepath = r'd:/CNTDATA/CNTA_ML_Project/index.html'

with open(filepath, 'r', encoding='utf-8') as f:
    content = f.read()

viewer_stage = """                <!-- Center: Viewing Stage -->
                <div id="clean-viewer-stage" class="flex-1 flex flex-col min-w-0 border-r border-slate-200 bg-slate-100 relative">
                    <!-- Main image pane -->
                    <div id="clean-image-panel" class="flex-1 flex flex-col min-h-0 relative p-4 group justify-center items-center">
                        <div class="absolute top-4 left-4 right-4 flex justify-between items-center z-10 pointer-events-none">
                            <div id="clean-current-label" class="pointer-events-auto px-4 py-2 bg-white/90 backdrop-blur rounded-lg border border-slate-200 font-black text-slate-700 text-[11px] shadow-sm">请在左侧选择样品</div>
                            <button id="clean-open-original" type="button" onclick="openCleanOriginal()" class="pointer-events-auto px-4 py-2 bg-white hover:bg-slate-50 text-blue-600 border border-slate-200 rounded-lg text-xs font-black transition-colors shadow-sm">
                                <i class="fas fa-expand mr-1.5"></i>放大图像
                            </button>
                        </div>
                        <img id="clean-original-image" class="hidden max-w-full max-h-full object-contain cursor-zoom-in rounded-lg shadow-sm bg-white" alt="大舞台原图" onclick="openCleanOriginal()">
                        <div id="clean-original-placeholder" class="absolute inset-0 flex items-center justify-center text-sm font-black text-slate-400 uppercase tracking-widest">尚未选择样品</div>
                    </div>
                    
                    <!-- Quick Note Strip under image -->
                    <div class="px-5 py-3 border-t border-b border-slate-200 bg-white/80 shrink-0 backdrop-blur">
                        <div class="flex items-center gap-3">
                            <span class="text-[10px] font-black text-blue-500 uppercase tracking-widest whitespace-nowrap"><i class="fas fa-bolt mr-1"></i>快速结论</span>
                            <div id="clean-quick-note" class="text-xs text-slate-600 w-full truncate font-bold">请选择样品查看分析摘要。</div>
                        </div>
                    </div>

                    <!-- Step Strip Toolbar -->
                    <div id="clean-step-strip" class="py-3 px-5 flex flex-col shrink-0 bg-white z-10">
                        <div class="flex justify-between items-center shrink-0 mb-2">
                            <span class="text-[11px] font-black text-slate-500 uppercase tracking-widest">处理步骤视图切换</span>
                            <span class="text-[10px] text-slate-400">点击按钮在上方主区域显示对应步骤的图像</span>
                        </div>
                        <div id="clean-step-grid" class="flex flex-wrap gap-2">
                        </div>
                    </div>
                </div>
"""

# Pattern to find the clean-review-panel block
# It might have flex-1 if it's the 2nd column now
pattern = r'(<!--\s*Review Panel:.*?\s*-->\s*<div\s+id="clean-review-panel"\s+class=")(flex-1)(.*?>)'

if re.search(pattern, content):
    print("Found review panel with flex-1. Inserting viewer stage and shrinking panel.")
    new_content = re.sub(pattern, viewer_stage + r'\1w-[360px]\3', content)
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(new_content)
    print("Success.")
else:
    print("Could not find the review panel with expected classes.")
    # Fallback to a simpler match if needed
    if 'id="clean-review-panel"' in content:
        print("Found id='clean-review-panel'. Forcing insertion.")
        new_content = content.replace('id="clean-review-panel"', viewer_stage + 'id="clean-review-panel"')
        new_content = new_content.replace('id="clean-review-panel" class="flex-1', 'id="clean-review-panel" class="w-[360px]')
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(new_content)
        print("Success (forced).")
    else:
        print("Fatal: id='clean-review-panel' not found.")
