echo "# Optuna Heterogeneous Dual Branch Network" > README.md
echo "## 项目描述" >> README.md
echo "基于深度学习的调制信号分类识别系统" >> README.md
echo "### 包含的模型：" >> README.md
echo "- DAE" >> README.md
echo "- MCLDNN" >> README.md
echo "- PET-CGDNN" >> README.md
echo "- WACN" >> README.md

git add README.md
git commit -m "添加项目README文件"
git push
