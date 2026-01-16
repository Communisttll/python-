// 主要JavaScript功能

// 文件上传处理
document.addEventListener('DOMContentLoaded', function() {
    const uploadArea = document.getElementById('uploadArea');
    const fileInput = document.getElementById('fileInput');
    const previewContainer = document.getElementById('imagePreview');
    const uploadForm = document.getElementById('uploadForm');
    
    if (uploadArea && fileInput) {
        // 点击上传区域触发文件选择
        uploadArea.addEventListener('click', function() {
            fileInput.click();
        });
        
        // 拖拽上传
        uploadArea.addEventListener('dragover', function(e) {
            e.preventDefault();
            uploadArea.classList.add('dragover');
        });
        
        uploadArea.addEventListener('dragleave', function(e) {
            e.preventDefault();
            uploadArea.classList.remove('dragover');
        });
        
        uploadArea.addEventListener('drop', function(e) {
            e.preventDefault();
            uploadArea.classList.remove('dragover');
            
            const files = e.dataTransfer.files;
            if (files.length > 0 && files[0].type.startsWith('image/')) {
                handleFileSelect(files[0]);
            }
        });
        
        // 文件选择处理
        fileInput.addEventListener('change', function(e) {
            if (e.target.files.length > 0) {
                handleFileSelect(e.target.files[0]);
            }
        });
    }
    
    // 处理文件选择
    function handleFileSelect(file) {
        if (!file.type.startsWith('image/')) {
            alert('请选择有效的图片文件！');
            return;
        }
        
        // 显示预览
        const reader = new FileReader();
        reader.onload = function(e) {
            const img = document.createElement('img');
            img.src = e.target.result;
            img.className = 'image-preview';
            
            previewContainer.innerHTML = '';
            previewContainer.appendChild(img);
            
            // 显示上传按钮
            const uploadBtn = document.getElementById('uploadBtn');
            if (uploadBtn) {
                uploadBtn.style.display = 'inline-block';
            }
        };
        reader.readAsDataURL(file);
    }
    
    // 表单提交处理
    if (uploadForm) {
        uploadForm.addEventListener('submit', function(e) {
            e.preventDefault();
            
            const file = fileInput.files[0];
            if (!file) {
                alert('请选择要上传的图片！');
                return;
            }
            
            // 显示加载状态
            showLoading();
            
            // 创建FormData并提交
            const formData = new FormData();
            formData.append('image', file);
            
            fetch(uploadForm.action, {
                method: 'POST',
                body: formData,
                headers: {
                    'X-CSRFToken': getCookie('csrftoken')
                }
            })
            .then(response => response.json())
            .then(data => {
                hideLoading();
                if (data.success) {
                    displayResults(data);
                } else {
                    alert('处理失败：' + data.error);
                }
            })
            .catch(error => {
                hideLoading();
                console.error('Error:', error);
                alert('上传失败，请重试！');
            });
        });
    }
    
    // 显示加载状态
    function showLoading() {
        const loadingDiv = document.getElementById('loadingSpinner');
        const resultsDiv = document.getElementById('resultsContainer');
        
        if (loadingDiv) {
            loadingDiv.style.display = 'block';
        }
        if (resultsDiv) {
            resultsDiv.style.display = 'none';
        }
    }
    
    // 隐藏加载状态
    function hideLoading() {
        const loadingDiv = document.getElementById('loadingSpinner');
        const resultsDiv = document.getElementById('resultsContainer');
        
        if (loadingDiv) {
            loadingDiv.style.display = 'none';
        }
        if (resultsDiv) {
            resultsDiv.style.display = 'block';
        }
    }
    
    // 显示结果
    function displayResults(data) {
        const resultsContainer = document.getElementById('resultsContainer');
        if (!resultsContainer) return;
        
        // 清空之前的结果
        resultsContainer.innerHTML = '';
        
        // 显示分析信息
        if (data.analysis) {
            const analysisDiv = createAnalysisCard(data.analysis);
            resultsContainer.appendChild(analysisDiv);
        }
        
        // 显示本地图库结果
        if (data.local_results && data.local_results.length > 0) {
            const localResultsDiv = createResultsSection('本地图库结果', data.local_results, 'local');
            resultsContainer.appendChild(localResultsDiv);
        }
        

        
        // 显示特征可视化
        if (data.features) {
            const featuresDiv = createFeatureVisualization(data.features);
            resultsContainer.appendChild(featuresDiv);
        }
    }
    
    // 创建分析卡片
    function createAnalysisCard(analysis) {
        const div = document.createElement('div');
        div.className = 'col-12';
        div.innerHTML = `
            <div class="card analysis-card">
                <div class="card-body">
                    <h5 class="card-title"><i class="fas fa-chart-line"></i> 图像分析</h5>
                    <div class="row">
                        <div class="col-md-3">
                            <div class="text-center">
                                <h3 class="text-primary">${analysis.feature_dim}</h3>
                                <p class="text-muted">特征维度</p>
                            </div>
                        </div>
                        <div class="col-md-3">
                            <div class="text-center">
                                <h3 class="text-success">${analysis.processing_time}</h3>
                                <p class="text-muted">处理时间(秒)</p>
                            </div>
                        </div>
                        <div class="col-md-3">
                            <div class="text-center">
                                <h3 class="text-warning">${analysis.local_matches}</h3>
                                <p class="text-muted">本地匹配数</p>
                            </div>
                        </div>
                    </div>
                </div>
            </div>
        `;
        return div;
    }
    
    // 创建结果区域
    function createResultsSection(title, results, type) {
        const div = document.createElement('div');
        div.className = 'col-12 results-section';
        
        let html = `
            <h4 class="section-title">${title}</h4>
            <div class="row">
        `;
        
        results.forEach((result, index) => {
            html += `
                <div class="col-lg-3 col-md-4 col-sm-6 mb-4">
                    <div class="card result-card">
                        <div class="position-relative">
                            <img src="${result.image_url}" class="card-img-top" alt="相似图片">
                            <div class="similarity-badge">
                                ${(result.similarity * 100).toFixed(1)}% 相似
                            </div>
                        </div>
                        <div class="card-body">
                            <p class="card-text">
                                <small class="text-muted">
                                    <i class="fas fa-tag"></i> ${result.caption || '无描述'}
                                </small>
                            </p>
                            ${result.source ? `<p class="card-text">
                                <small class="text-muted">
                                    <i class="fas fa-globe"></i> 来源: ${result.source}
                                </small>
                            </p>` : ''}
                        </div>
                    </div>
                </div>
            `;
        });
        
        html += '</div>';
        div.innerHTML = html;
        return div;
    }
    
    // 创建特征可视化
    function createFeatureVisualization(features) {
        const div = document.createElement('div');
        div.className = 'col-12';
        
        let html = `
            <div class="card">
                <div class="card-body">
                    <h5 class="card-title"><i class="fas fa-chart-bar"></i> 特征向量可视化</h5>
                    <p class="text-muted">768维特征向量的前50维展示</p>
                    <div class="feature-visualization">
        `;
        
        // 显示前50维特征
        const topFeatures = features.slice(0, 50);
        const maxVal = Math.max(...topFeatures.map(Math.abs));
        
        topFeatures.forEach((value, index) => {
            const width = (Math.abs(value) / maxVal) * 100;
            const color = value >= 0 ? '#007bff' : '#dc3545';
            html += `
                <div class="d-flex align-items-center mb-1">
                    <span class="me-2" style="width: 30px;">${index + 1}</span>
                    <div class="feature-bar" style="width: ${width}%; background-color: ${color};"></div>
                    <span class="ms-2" style="width: 60px;">${value.toFixed(3)}</span>
                </div>
            `;
        });
        
        html += `
                    </div>
                </div>
            </div>
        `;
        
        div.innerHTML = html;
        return div;
    }
    
    // 获取CSRF Token
    function getCookie(name) {
        let cookieValue = null;
        if (document.cookie && document.cookie !== '') {
            const cookies = document.cookie.split(';');
            for (let i = 0; i < cookies.length; i++) {
                const cookie = cookies[i].trim();
                if (cookie.substring(0, name.length + 1) === (name + '=')) {
                    cookieValue = decodeURIComponent(cookie.substring(name.length + 1));
                    break;
                }
            }
        }
        return cookieValue;
    }
});