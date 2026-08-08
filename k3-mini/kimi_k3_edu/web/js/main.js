/* ============================================
   K3-Edu Interactive Documentation
   Three.js, GSAP, Canvas animations
   ============================================ */

document.addEventListener('DOMContentLoaded', () => {
    initHeroCanvas();
    initNavigation();
    initScrollAnimations();
    initCounterAnimations();
    initParameterViz();
    initAttentionViz();
    initTimelineAnimations();
    initSamplingViz();
    initCodebaseExplorer();
    initLayerExplorer();
    initTiltCards();
});

/* ============================================
   Hero 3D Canvas - Neural Network Visualization
   ============================================ */

function initHeroCanvas() {
    const canvas = document.getElementById('heroCanvas');
    if (!canvas) return;

    const scene = new THREE.Scene();
    const camera = new THREE.PerspectiveCamera(75, window.innerWidth / window.innerHeight, 0.1, 1000);
    const renderer = new THREE.WebGLRenderer({ canvas, alpha: true, antialias: true });

    renderer.setSize(window.innerWidth, window.innerHeight);
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));

    // Create neural network nodes
    const nodeCount = 80;
    const nodes = [];
    const nodeGeometry = new THREE.SphereGeometry(0.08, 16, 16);

    for (let i = 0; i < nodeCount; i++) {
        const material = new THREE.MeshBasicMaterial({
            color: new THREE.Color().setHSL(0.45 + Math.random() * 0.15, 0.8, 0.6),
            transparent: true,
            opacity: 0.6 + Math.random() * 0.4
        });

        const node = new THREE.Mesh(nodeGeometry, material);
        node.position.set(
            (Math.random() - 0.5) * 12,
            (Math.random() - 0.5) * 8,
            (Math.random() - 0.5) * 6
        );

        node.userData = {
            velocity: new THREE.Vector3(
                (Math.random() - 0.5) * 0.01,
                (Math.random() - 0.5) * 0.01,
                (Math.random() - 0.5) * 0.005
            ),
            originalPos: node.position.clone(),
            phase: Math.random() * Math.PI * 2
        };

        scene.add(node);
        nodes.push(node);
    }

    // Create connections
    const lineMaterial = new THREE.LineBasicMaterial({
        color: 0x00d4aa,
        transparent: true,
        opacity: 0.1
    });

    const connections = [];
    for (let i = 0; i < nodes.length; i++) {
        for (let j = i + 1; j < nodes.length; j++) {
            const dist = nodes[i].position.distanceTo(nodes[j].position);
            if (dist < 3) {
                const geometry = new THREE.BufferGeometry().setFromPoints([
                    nodes[i].position,
                    nodes[j].position
                ]);
                const line = new THREE.Line(geometry, lineMaterial.clone());
                line.userData = { nodeA: i, nodeB: j, maxDist: 3 };
                scene.add(line);
                connections.push(line);
            }
        }
    }

    camera.position.z = 8;

    // Mouse interaction
    let mouseX = 0, mouseY = 0;
    document.addEventListener('mousemove', (e) => {
        mouseX = (e.clientX / window.innerWidth - 0.5) * 2;
        mouseY = (e.clientY / window.innerHeight - 0.5) * 2;
    });

    // Animation loop
    let time = 0;
    function animate() {
        requestAnimationFrame(animate);
        time += 0.01;

        // Update nodes
        nodes.forEach((node, i) => {
            const data = node.userData;

            // Gentle floating motion
            node.position.x = data.originalPos.x + Math.sin(time + data.phase) * 0.3;
            node.position.y = data.originalPos.y + Math.cos(time * 0.7 + data.phase) * 0.2;
            node.position.z = data.originalPos.z + Math.sin(time * 0.5 + data.phase) * 0.15;

            // Mouse influence
            const dx = mouseX * 3 - node.position.x;
            const dy = -mouseY * 2 - node.position.y;
            node.position.x += dx * 0.002;
            node.position.y += dy * 0.002;

            // Pulse effect
            const scale = 1 + Math.sin(time * 2 + data.phase) * 0.2;
            node.scale.setScalar(scale);
        });

        // Update connections
        connections.forEach(line => {
            const nodeA = nodes[line.userData.nodeA];
            const nodeB = nodes[line.userData.nodeB];
            const positions = line.geometry.attributes.position.array;

            positions[0] = nodeA.position.x;
            positions[1] = nodeA.position.y;
            positions[2] = nodeA.position.z;
            positions[3] = nodeB.position.x;
            positions[4] = nodeB.position.y;
            positions[5] = nodeB.position.z;

            line.geometry.attributes.position.needsUpdate = true;

            const dist = nodeA.position.distanceTo(nodeB.position);
            line.material.opacity = Math.max(0, 0.15 * (1 - dist / line.userData.maxDist));
        });

        // Camera gentle rotation
        camera.position.x = Math.sin(time * 0.1) * 0.5;
        camera.position.y = Math.cos(time * 0.15) * 0.3;
        camera.lookAt(0, 0, 0);

        renderer.render(scene, camera);
    }

    animate();

    // Resize handler
    window.addEventListener('resize', () => {
        camera.aspect = window.innerWidth / window.innerHeight;
        camera.updateProjectionMatrix();
        renderer.setSize(window.innerWidth, window.innerHeight);
    });
}

/* ============================================
   Navigation
   ============================================ */

function initNavigation() {
    const nav = document.getElementById('nav');
    const toggle = document.getElementById('navToggle');
    const links = document.querySelector('.nav-links');

    // Scroll effect
    window.addEventListener('scroll', () => {
        if (window.scrollY > 100) {
            nav.classList.add('scrolled');
        } else {
            nav.classList.remove('scrolled');
        }
    });

    // Mobile toggle
    if (toggle) {
        toggle.addEventListener('click', () => {
            links.classList.toggle('active');
        });
    }

    // Smooth scroll for nav links
    document.querySelectorAll('.nav-link').forEach(link => {
        link.addEventListener('click', (e) => {
            e.preventDefault();
            const target = document.querySelector(link.getAttribute('href'));
            if (target) {
                target.scrollIntoView({ behavior: 'smooth' });
                links.classList.remove('active');
            }
        });
    });
}

/* ============================================
   Scroll Animations with GSAP
   ============================================ */

function initScrollAnimations() {
    gsap.registerPlugin(ScrollTrigger);

    // Hero entrance animations
    gsap.to('.hero-badge', {
        opacity: 1,
        y: 0,
        duration: 0.8,
        delay: 0.3,
        ease: 'power3.out'
    });

    document.querySelectorAll('.hero-line').forEach((line, i) => {
        gsap.to(line, {
            opacity: 1,
            y: 0,
            rotateX: 0,
            duration: 1,
            delay: 0.5 + i * 0.15,
            ease: 'power3.out'
        });
    });

    gsap.to('.hero-desc', {
        opacity: 1,
        y: 0,
        duration: 0.8,
        delay: 1,
        ease: 'power3.out'
    });

    document.querySelectorAll('.stat-item').forEach((item, i) => {
        gsap.to(item, {
            opacity: 1,
            y: 0,
            duration: 0.6,
            delay: 1.2 + i * 0.1,
            ease: 'power3.out'
        });
    });

    gsap.to('.hero-actions', {
        opacity: 1,
        y: 0,
        duration: 0.6,
        delay: 1.6,
        ease: 'power3.out'
    });

    gsap.to('.hero-scroll', {
        opacity: 1,
        duration: 0.6,
        delay: 2,
        ease: 'power3.out'
    });

    // Section headers
    document.querySelectorAll('.section-header').forEach(header => {
        gsap.from(header, {
            scrollTrigger: {
                trigger: header,
                start: 'top 80%',
                toggleActions: 'play none none none'
            },
            opacity: 0,
            y: 40,
            duration: 0.8,
            ease: 'power3.out'
        });
    });

    // Architecture cards
    document.querySelectorAll('.arch-card').forEach((card, i) => {
        gsap.from(card, {
            scrollTrigger: {
                trigger: card,
                start: 'top 85%',
                toggleActions: 'play none none none'
            },
            opacity: 0,
            y: 50,
            rotateX: 10,
            duration: 0.7,
            delay: i * 0.1,
            ease: 'power3.out'
        });
    });

    // Timeline phases
    document.querySelectorAll('.timeline-phase').forEach((phase, i) => {
        gsap.to(phase, {
            scrollTrigger: {
                trigger: phase,
                start: 'top 80%',
                toggleActions: 'play none none none'
            },
            opacity: 1,
            x: 0,
            duration: 0.8,
            delay: i * 0.15,
            ease: 'power3.out'
        });
    });

    // Optimization cards
    document.querySelectorAll('.opt-card').forEach((card, i) => {
        gsap.from(card, {
            scrollTrigger: {
                trigger: card,
                start: 'top 85%',
                toggleActions: 'play none none none'
            },
            opacity: 0,
            y: 30,
            duration: 0.6,
            delay: i * 0.08,
            ease: 'power3.out'
        });
    });
}

/* ============================================
   Counter Animations
   ============================================ */

function initCounterAnimations() {
    const counters = document.querySelectorAll('.stat-value[data-count]');

    const observer = new IntersectionObserver((entries) => {
        entries.forEach(entry => {
            if (entry.isIntersecting) {
                const el = entry.target;
                const target = parseInt(el.dataset.count);
                animateCounter(el, target);
                observer.unobserve(el);
            }
        });
    }, { threshold: 0.5 });

    counters.forEach(counter => observer.observe(counter));
}

function animateCounter(el, target) {
    const duration = 2000;
    const start = performance.now();

    function update(now) {
        const elapsed = now - start;
        const progress = Math.min(elapsed / duration, 1);
        const eased = 1 - Math.pow(1 - progress, 3);
        const current = Math.floor(eased * target);

        el.textContent = current.toLocaleString();

        if (progress < 1) {
            requestAnimationFrame(update);
        } else {
            el.textContent = target.toLocaleString();
        }
    }

    requestAnimationFrame(update);
}

/* ============================================
   Parameter Visualization (Canvas)
   ============================================ */

function initParameterViz() {
    const canvas = document.getElementById('paramCanvas');
    if (!canvas) return;

    const ctx = canvas.getContext('2d');
    canvas.width = 300;
    canvas.height = 300;

    const params = [
        { name: 'Embedding', value: 38.4, color: '#00d4aa' },
        { name: 'Attention', value: 28.8, color: '#6366f1' },
        { name: 'FFN', value: 85.2, color: '#f59e0b' },
        { name: 'Norm + Output', value: 47.6, color: '#ec4899' }
    ];

    const total = params.reduce((sum, p) => sum + p.value, 0);
    let currentAngle = -Math.PI / 2;

    // Animate drawing
    let progress = 0;
    function draw() {
        progress = Math.min(progress + 0.02, 1);

        ctx.clearRect(0, 0, canvas.width, canvas.height);

        const centerX = canvas.width / 2;
        const centerY = canvas.height / 2;
        const radius = 100;
        const innerRadius = 60;

        let angle = -Math.PI / 2;

        params.forEach(param => {
            const sliceAngle = (param.value / total) * Math.PI * 2 * progress;

            // Draw slice
            ctx.beginPath();
            ctx.arc(centerX, centerY, radius, angle, angle + sliceAngle);
            ctx.arc(centerX, centerY, innerRadius, angle + sliceAngle, angle, true);
            ctx.closePath();

            ctx.fillStyle = param.color;
            ctx.globalAlpha = 0.8;
            ctx.fill();

            ctx.strokeStyle = param.color;
            ctx.lineWidth = 2;
            ctx.globalAlpha = 1;
            ctx.stroke();

            // Glow effect
            ctx.shadowColor = param.color;
            ctx.shadowBlur = 20;
            ctx.stroke();
            ctx.shadowBlur = 0;

            angle += sliceAngle;
        });

        // Center text
        ctx.fillStyle = '#e8e8f0';
        ctx.font = 'bold 24px "JetBrains Mono"';
        ctx.textAlign = 'center';
        ctx.textBaseline = 'middle';
        ctx.fillText('200M', centerX, centerY - 8);

        ctx.fillStyle = '#a0a0b8';
        ctx.font = '12px "Space Grotesk"';
        ctx.fillText('Parameters', centerX, centerY + 12);

        if (progress < 1) {
            requestAnimationFrame(draw);
        }
    }

    // Trigger on scroll
    const observer = new IntersectionObserver((entries) => {
        if (entries[0].isIntersecting) {
            draw();
            observer.disconnect();
        }
    });

    observer.observe(canvas);
}

/* ============================================
   Attention Visualization
   ============================================ */

function initAttentionViz() {
    const canvas = document.getElementById('kdaCanvas');
    if (!canvas) return;

    const ctx = canvas.getContext('2d');
    canvas.width = canvas.offsetWidth || 500;
    canvas.height = 250;

    const seqLenSlider = document.getElementById('seqLenSlider');
    const headDimSlider = document.getElementById('headDimSlider');
    const betaSlider = document.getElementById('betaSlider');

    let seqLen = 8;
    let headDim = 16;
    let beta = 0.5;
    let time = 0;

    if (seqLenSlider) {
        seqLenSlider.addEventListener('input', (e) => {
            seqLen = parseInt(e.target.value);
            document.getElementById('seqLenValue').textContent = seqLen;
        });
    }

    if (headDimSlider) {
        headDimSlider.addEventListener('input', (e) => {
            headDim = parseInt(e.target.value);
            document.getElementById('headDimValue').textContent = headDim;
        });
    }

    if (betaSlider) {
        betaSlider.addEventListener('input', (e) => {
            beta = parseInt(e.target.value) / 100;
            document.getElementById('betaValue').textContent = beta.toFixed(2);
        });
    }

    function drawKDA() {
        ctx.clearRect(0, 0, canvas.width, canvas.height);

        const cellSize = Math.min(canvas.width / (seqLen + 2), 30);
        const startX = (canvas.width - seqLen * cellSize) / 2;
        const startY = 40;

        // Draw memory state matrix
        ctx.fillStyle = '#1a1a25';
        ctx.fillRect(startX - 5, startY - 5, seqLen * cellSize + 10, seqLen * cellSize + 10);

        for (let i = 0; i < seqLen; i++) {
            for (let j = 0; j < seqLen; j++) {
                const value = Math.sin(time + i * 0.5 + j * 0.3) * 0.5 + 0.5;
                const alpha = i <= j ? value * beta : 0.1;

                ctx.fillStyle = `rgba(0, 212, 170, ${alpha})`;
                ctx.fillRect(startX + j * cellSize, startY + i * cellSize, cellSize - 2, cellSize - 2);
            }
        }

        // Labels
        ctx.fillStyle = '#a0a0b8';
        ctx.font = '10px "JetBrains Mono"';
        ctx.textAlign = 'center';

        for (let i = 0; i < seqLen; i++) {
            ctx.fillText(`t${i}`, startX + i * cellSize + cellSize / 2, startY - 10);
        }

        ctx.textAlign = 'right';
        for (let i = 0; i < seqLen; i++) {
            ctx.fillText(`t${i}`, startX - 10, startY + i * cellSize + cellSize / 2 + 4);
        }

        // Title
        ctx.textAlign = 'center';
        ctx.fillStyle = '#e8e8f0';
        ctx.font = '12px "Space Grotesk"';
        ctx.fillText('Memory State S (causal mask applied)', canvas.width / 2, 20);

        time += 0.05;
        requestAnimationFrame(drawKDA);
    }

    drawKDA();
}

/* ============================================
   Timeline Animations
   ============================================ */

function initTimelineAnimations() {
    // LR schedule canvas
    const lrCanvas = document.getElementById('lrCanvas1');
    if (!lrCanvas) return;

    const ctx = lrCanvas.getContext('2d');
    lrCanvas.width = lrCanvas.offsetWidth || 400;
    lrCanvas.height = 100;

    function drawLRSchedule() {
        ctx.clearRect(0, 0, lrCanvas.width, lrCanvas.height);

        const steps = 150000;
        const warmup = 2000;
        const w = lrCanvas.width;
        const h = lrCanvas.height;

        ctx.beginPath();
        ctx.strokeStyle = '#00d4aa';
        ctx.lineWidth = 2;

        for (let x = 0; x < w; x++) {
            const step = (x / w) * steps;
            let lr;

            if (step < warmup) {
                lr = 1e-5 + (1e-3 - 1e-5) * (step / warmup);
            } else if (step < 50000) {
                const p = (step - warmup) / (50000 - warmup);
                lr = 1e-3 - (1e-3 - 5e-4) * p;
            } else {
                const p = (step - 50000) / (steps - 50000);
                lr = 5e-4 - (5e-4 - 1e-5) * (0.5 * (1 + Math.cos(Math.PI * p)));
            }

            const y = h - (lr / 1e-3) * (h - 20) - 10;

            if (x === 0) ctx.moveTo(x, y);
            else ctx.lineTo(x, y);
        }

        ctx.stroke();

        // Glow
        ctx.shadowColor = '#00d4aa';
        ctx.shadowBlur = 10;
        ctx.stroke();
        ctx.shadowBlur = 0;

        // Labels
        ctx.fillStyle = '#6b6b80';
        ctx.font = '10px "JetBrains Mono"';
        ctx.fillText('1e-3', 5, 15);
        ctx.fillText('1e-5', 5, h - 5);
    }

    drawLRSchedule();
}

/* ============================================
   Sampling Visualization
   ============================================ */

function initSamplingViz() {
    const canvas = document.getElementById('samplingCanvas');
    if (!canvas) return;

    const ctx = canvas.getContext('2d');
    canvas.width = canvas.offsetWidth || 400;
    canvas.height = 150;

    let temperature = 0.7;
    const buttons = document.querySelectorAll('.sampling-btn');

    buttons.forEach(btn => {
        btn.addEventListener('click', () => {
            buttons.forEach(b => b.classList.remove('active'));
            btn.classList.add('active');
            temperature = parseFloat(btn.dataset.temp);
        });
    });

    function drawDistribution() {
        ctx.clearRect(0, 0, canvas.width, canvas.height);

        const w = canvas.width;
        const h = canvas.height;
        const nTokens = 20;
        const barWidth = (w - 40) / nTokens;

        // Generate sample logits
        const logits = [];
        for (let i = 0; i < nTokens; i++) {
            logits.push(Math.random() * 2 - 1);
        }

        // Apply temperature
        const scaled = logits.map(l => l / temperature);
        const maxLogit = Math.max(...scaled);
        const expSum = scaled.reduce((sum, l) => sum + Math.exp(l - maxLogit), 0);
        const probs = scaled.map(l => Math.exp(l - maxLogit) / expSum);

        // Draw bars
        const maxProb = Math.max(...probs);

        for (let i = 0; i < nTokens; i++) {
            const barHeight = (probs[i] / maxProb) * (h - 40);
            const x = 20 + i * barWidth;
            const y = h - 20 - barHeight;

            // Color based on probability
            const intensity = probs[i] / maxProb;
            ctx.fillStyle = `rgba(0, 212, 170, ${0.3 + intensity * 0.7})`;

            ctx.fillRect(x + 2, y, barWidth - 4, barHeight);

            // Highlight top token
            if (i === probs.indexOf(maxProb)) {
                ctx.strokeStyle = '#00d4aa';
                ctx.lineWidth = 2;
                ctx.strokeRect(x + 1, y - 1, barWidth - 2, barHeight + 2);
            }
        }

        // Labels
        ctx.fillStyle = '#6b6b80';
        ctx.font = '10px "JetBrains Mono"';
        ctx.textAlign = 'center';
        ctx.fillText(`Temperature: ${temperature}`, w / 2, 15);

        requestAnimationFrame(drawDistribution);
    }

    drawDistribution();
}

/* ============================================
   Codebase Explorer
   ============================================ */

function initCodebaseExplorer() {
    const files = document.querySelectorAll('.tree-file');
    const codeDisplay = document.getElementById('codeDisplay');
    const codeExplanation = document.getElementById('codeExplanation');

    const fileContents = {
        'config.h': {
            code: `/**
 * config.h - Hyperparameters and configuration
 * 
 * Inspired by MoonshotAI's Kimi K3 architecture
 */

#ifndef CONFIG_H
#define CONFIG_H

#define MODEL_NAME          "K3-Edu-200M"
#define D_MODEL             768
#define N_LAYERS            12
#define N_HEADS             12
#define HEAD_DIM            64
#define D_FFN               3072
#define MAX_SEQ_LEN         8192

/* Training hyperparameters */
#define BASE_BATCH_SIZE     4
#define BASE_PEAK_LR        3e-4f
#define BASE_WARMUP_STEPS   2000
#define BASE_TOTAL_STEPS    150000

/* KDA parameters */
#define KDA_CHUNK_SIZE      128

/* Special tokens */
#define SPECIAL_IM_START    "<|im_start|>"
#define SPECIAL_IM_END      "|>"

#endif /* CONFIG_H */`,
            explanation: 'Centralizes all hyperparameters. Makes experimentation easy without hunting through the codebase.',
            tags: ['Architecture', 'Hyperparameters', 'Configuration']
        },
        'tokenizer.h': {
            code: `/**
 * tokenizer.h - Byte Pair Encoding Tokenizer
 * 
 * GPT-2 style pretokenization with special instruction tokens
 */

#ifndef TOKENIZER_H
#define TOKENIZER_H

enum SpecialTokenID {
    TOK_PAD = 0, TOK_UNK = 1, TOK_BOS = 2,
    TOK_EOS = 3, TOK_MASK = 4, TOK_IM_START = 5,
    TOK_IM_END = 6, TOK_SYSTEM = 7,
    TOK_USER = 8, TOK_ASSISTANT = 9
};

typedef struct {
    char* text;
    int id;
    int freq;
} Token;

Tokenizer* tokenizer_create(int initial_vocab_size);
int tokenizer_train(Tokenizer* tok, const char* dataset_dir, int target_vocab_size);
int* tokenizer_encode(Tokenizer* tok, const char* text, int* out_len);
char* tokenizer_decode(Tokenizer* tok, const int* ids, int len);

#endif`,
            explanation: 'BPE tokenizer with 10 special tokens for instruction tuning. Supports multilingual text and code.',
            tags: ['Tokenization', 'BPE', 'Special Tokens']
        },
        'model.h': {
            code: `/**
 * model.h - Transformer Model Architecture
 * 
 * KDA + AttnRes + RMSNorm + RoPE + SwiGLU
 */

typedef struct {
    Tensor* w_q, *w_k, *w_v, *w_o;
    Tensor* w_gate, *w_beta;
    RMSNorm* norm;
    int n_heads, head_dim, d_model;
} KDAAttention;

typedef struct {
    Tensor* w_gate, *w_up, *w_down;
    RMSNorm* norm;
    int d_model, d_ffn;
} FFN;

typedef struct {
    Tensor* token_embedding;
    TransformerLayer** layers;
    RMSNorm* final_norm;
    RoPE* rope;
    AttnRes* attnres;
    int d_model, n_layers, vocab_size;
} TransformerModel;

TransformerModel* model_create(const Config* cfg, Tokenizer* tokenizer);
Tensor* model_forward(TransformerModel* model, const int* token_ids, 
                      int batch, int seq_len, int training);`,
            explanation: 'Core model structure. KDA attention replaces standard softmax attention for O(1) memory per token.',
            tags: ['Transformer', 'KDA', 'Architecture']
        },
        'train.h': {
            code: `/**
 * train.h - Training Infrastructure
 * 
 * Three-phase training with all optimizations
 */

typedef struct {
    float peak_lr, min_lr;
    int warmup_steps, current_step, total_steps;
    int restart_period, n_restarts;
} LRScheduler;

typedef struct {
    TransformerModel* model;
    AdamWOptimizer* optimizer;
    LRScheduler* scheduler;
    float train_loss, val_loss, best_val_loss;
    int grad_accum_steps, plateau_count;
    int global_step, epoch;
} TrainState;

int train_base(TrainState* state, Dataset* dataset, const Config* cfg);
int train_instruction(TrainState* state, Dataset* dataset, const Config* cfg);
float train_step(TrainState* state, const int* input_ids, 
                 const int* target_ids, int batch_size, int seq_len);`,
            explanation: 'Training state management with AdamW, cosine+warm restarts LR, gradient accumulation, and plateau detection.',
            tags: ['Training', 'AdamW', 'Scheduler']
        },
        'inference.h': {
            code: `/**
 * inference.h - Inference Engine
 * 
 * CPU + GPU execution with sampling and KV-cache
 */

typedef struct {
    float temperature;
    int top_k;
    float top_p;
    int max_new_tokens;
    float repetition_penalty;
    int do_sample;
} GenerationConfig;

int generate(TransformerModel* model, Tokenizer* tokenizer,
             const char* prompt, const GenerationConfig* config,
             char* out_text, int out_max_len);

char* chat_completion(TransformerModel* model, Tokenizer* tokenizer,
                      const ChatMessage* messages, int n_messages,
                      const GenerationConfig* config);

float evaluate_perplexity(TransformerModel* model, Tokenizer* tokenizer,
                          const char* text);`,
            explanation: 'Text generation with temperature, top-k, top-p filtering. Chat API with special token formatting.',
            tags: ['Inference', 'Sampling', 'Chat']
        }
    };

    files.forEach(file => {
        file.addEventListener('click', () => {
            files.forEach(f => f.classList.remove('active'));
            file.classList.add('active');

            const fileName = file.dataset.file;
            const content = fileContents[fileName];

            if (content && codeDisplay) {
                codeDisplay.textContent = content.code;
                Prism.highlightElement(codeDisplay);
            }

            if (content && codeExplanation) {
                codeExplanation.querySelector('h4').textContent = 'What this file does';
                codeExplanation.querySelector('p').textContent = content.explanation;

                const tagsContainer = codeExplanation.querySelector('.explanation-tags');
                tagsContainer.innerHTML = content.tags.map(tag => 
                    `<span class="tag">${tag}</span>`
                ).join('');
            }
        });
    });

    // Folder toggle
    document.querySelectorAll('.tree-folder > .folder-name, .tree-folder > .folder-icon').forEach(folder => {
        folder.addEventListener('click', () => {
            const parent = folder.parentElement;
            parent.classList.toggle('expanded');
        });
    });
}

/* ============================================
   Layer Explorer
   ============================================ */

function initLayerExplorer() {
    const layerBoxes = document.querySelectorAll('.layer-box');
    const layerInfo = document.getElementById('layerInfo');
    const progressBar = document.getElementById('progressBar');
    let currentLayer = 0;

    const layerDescriptions = {
        'Input': 'Raw token IDs enter the model. Shape: [Batch, Sequence]',
        'Token Embedding': 'Each token ID is mapped to a 768-dimensional vector. This is a lookup table.',
        'RMSNorm': 'Root-mean-square normalization. Scales inputs without subtracting mean.',
        'KDA Attention': 'Kimi Delta Attention. Linear attention with channel-wise gating. O(1) memory per token.',
        'SwiGLU FFN': 'Feed-forward network with gating. gate(x) * silu(up(x)) -> down projection.',
        'Output': 'Final projection to vocabulary space. Produces logits for each token.',
    };

    layerBoxes.forEach((box, index) => {
        box.addEventListener('click', () => {
            layerBoxes.forEach(b => b.classList.remove('active'));
            box.classList.add('active');

            const name = box.querySelector('.layer-name').textContent;
            if (layerInfo) {
                layerInfo.querySelector('h4').textContent = name;
                layerInfo.querySelector('p').textContent = layerDescriptions[name] || 'Layer component of the transformer.';
            }

            currentLayer = index;
            updateProgress();
        });

        box.addEventListener('mouseenter', () => {
            const name = box.querySelector('.layer-name').textContent;
            if (layerInfo && !box.classList.contains('active')) {
                layerInfo.querySelector('h4').textContent = name;
                layerInfo.querySelector('p').textContent = layerDescriptions[name] || 'Layer component of the transformer.';
            }
        });
    });

    function updateProgress() {
        if (progressBar) {
            const progress = ((currentLayer + 1) / layerBoxes.length) * 100;
            progressBar.style.width = progress + '%';
        }
    }

    // Navigation buttons
    const prevBtn = document.getElementById('prevLayer');
    const nextBtn = document.getElementById('nextLayer');
    const playBtn = document.getElementById('playLayer');
    let isPlaying = false;
    let playInterval;

    if (prevBtn) {
        prevBtn.addEventListener('click', () => {
            if (currentLayer > 0) {
                layerBoxes[currentLayer].classList.remove('active');
                currentLayer--;
                layerBoxes[currentLayer].classList.add('active');
                layerBoxes[currentLayer].click();
            }
        });
    }

    if (nextBtn) {
        nextBtn.addEventListener('click', () => {
            if (currentLayer < layerBoxes.length - 1) {
                layerBoxes[currentLayer].classList.remove('active');
                currentLayer++;
                layerBoxes[currentLayer].classList.add('active');
                layerBoxes[currentLayer].click();
            }
        });
    }

    if (playBtn) {
        playBtn.addEventListener('click', () => {
            if (isPlaying) {
                clearInterval(playInterval);
                playBtn.textContent = 'Play';
                isPlaying = false;
            } else {
                playBtn.textContent = 'Pause';
                isPlaying = true;
                currentLayer = 0;

                playInterval = setInterval(() => {
                    layerBoxes.forEach(b => b.classList.remove('active'));
                    layerBoxes[currentLayer].classList.add('active');
                    layerBoxes[currentLayer].click();

                    currentLayer++;
                    if (currentLayer >= layerBoxes.length) {
                        currentLayer = 0;
                    }
                }, 1500);
            }
        });
    }
}

/* ============================================
   3D Tilt Cards
   ============================================ */

function initTiltCards() {
    const cards = document.querySelectorAll('[data-tilt]');

    cards.forEach(card => {
        card.addEventListener('mousemove', (e) => {
            const rect = card.getBoundingClientRect();
            const x = e.clientX - rect.left;
            const y = e.clientY - rect.top;

            const centerX = rect.width / 2;
            const centerY = rect.height / 2;

            const rotateX = (y - centerY) / centerY * -8;
            const rotateY = (x - centerX) / centerX * 8;

            card.style.transform = `perspective(1000px) rotateX(${rotateX}deg) rotateY(${rotateY}deg) translateY(-4px)`;
        });

        card.addEventListener('mouseleave', () => {
            card.style.transform = 'perspective(1000px) rotateX(0) rotateY(0) translateY(0)';
        });
    });
}
