const state = {
  info: null,
  defaults: {},
};

const formatPercent = (value) => `${(Number(value || 0) * 100).toFixed(2)}%`;
const titleCase = (value) => String(value || '').replaceAll('_', ' ').replace(/\b\w/g, (c) => c.toUpperCase());

async function fetchJson(url, options = {}) {
  const response = await fetch(url, options);
  if (!response.ok) {
    throw new Error(`${url} returned ${response.status}`);
  }
  return response.json();
}

function createField(field) {
  const wrapper = document.createElement('div');
  wrapper.className = 'field';

  if (field.type === 'boolean') {
    wrapper.classList.add('toggle-row');
    wrapper.innerHTML = `
      <label for="${field.name}">${field.label}</label>
      <label class="switch" title="${field.label}">
        <input id="${field.name}" name="${field.name}" type="checkbox">
        <span></span>
      </label>
    `;
    return wrapper;
  }

  const label = document.createElement('label');
  label.setAttribute('for', field.name);
  label.textContent = field.label;
  wrapper.appendChild(label);

  if (field.type === 'select') {
    const select = document.createElement('select');
    select.id = field.name;
    select.name = field.name;
    field.options.forEach((option) => {
      const item = document.createElement('option');
      item.value = option;
      item.textContent = titleCase(option);
      select.appendChild(item);
    });
    wrapper.appendChild(select);
    return wrapper;
  }

  const input = document.createElement('input');
  input.id = field.name;
  input.name = field.name;
  input.type = 'number';
  input.step = 'any';
  if (field.min !== undefined) input.min = field.min;
  if (field.max !== undefined) input.max = field.max;
  wrapper.appendChild(input);
  return wrapper;
}

function setDefaults() {
  Object.entries(state.defaults).forEach(([name, value]) => {
    const input = document.querySelector(`[name="${name}"]`);
    if (!input) return;

    if (input.type === 'checkbox') {
      input.checked = Boolean(value);
    } else {
      input.value = value;
    }
  });
}

function collectPayload() {
  const payload = {};
  const form = document.getElementById('predictionForm');
  const data = new FormData(form);

  state.info.fields.forEach((field) => {
    const input = form.elements[field.name];
    if (!input) return;

    if (field.type === 'boolean') {
      payload[field.name] = input.checked ? 1 : 0;
    } else if (field.type === 'number') {
      payload[field.name] = Number(data.get(field.name));
    } else {
      payload[field.name] = data.get(field.name);
    }
  });

  return payload;
}

function renderMetrics(info) {
  document.getElementById('accuracyMetric').textContent = formatPercent(info.test_metrics.accuracy);
  document.getElementById('f1Metric').textContent = formatPercent(info.test_metrics.f1);
  document.getElementById('aucMetric').textContent = formatPercent(info.test_metrics.roc_auc);
  document.getElementById('modelName').textContent = titleCase(info.model_name);

  const table = document.getElementById('metricsTable');
  table.innerHTML = '';
  info.validation_metrics.forEach((row) => {
    const tr = document.createElement('tr');
    tr.innerHTML = `
      <td>${titleCase(row.model)}</td>
      <td>${formatPercent(row.accuracy)}</td>
      <td>${formatPercent(row.f1)}</td>
      <td>${formatPercent(row.roc_auc)}</td>
    `;
    table.appendChild(tr);
  });
}

function renderPipelineSteps(info) {
  const container = document.getElementById('pipelineSteps');
  container.innerHTML = '';
  info.pipeline_steps.forEach((step, index) => {
    const card = document.createElement('div');
    card.className = 'step-card';
    card.innerHTML = `<strong>${String(index + 1).padStart(2, '0')} ${step.title}</strong><p>${step.detail}</p>`;
    container.appendChild(card);
  });
}

function renderFeatureBars(info) {
  const container = document.getElementById('featureBars');
  container.innerHTML = '';
  const maxImportance = Math.max(...info.feature_importance.map((item) => Number(item.importance || 0)), 0.0001);

  info.feature_importance.forEach((item) => {
    const width = Math.max(2, (Number(item.importance || 0) / maxImportance) * 100);
    const row = document.createElement('div');
    row.className = 'feature-row';
    row.innerHTML = `
      <span title="${item.feature}">${item.feature}</span>
      <div class="bar-track"><div class="bar-fill" style="width:${width}%"></div></div>
      <small>${Number(item.importance || 0).toFixed(3)}</small>
    `;
    container.appendChild(row);
  });
}

function renderArtifacts(info) {
  const list = document.getElementById('artifactList');
  list.innerHTML = '';
  info.artifacts.forEach((artifact) => {
    const item = document.createElement('li');
    item.textContent = artifact;
    list.appendChild(item);
  });
}

function renderTrace(trace) {
  const list = document.getElementById('traceList');
  list.innerHTML = '';
  trace.forEach((step, index) => {
    const item = document.createElement('li');
    item.style.animationDelay = `${index * 90}ms`;
    item.innerHTML = `
      <span class="trace-dot"></span>
      <div>
        <strong>${step.name}</strong>
        <small>${step.value}</small>
      </div>
    `;
    list.appendChild(item);
  });
}

function renderPrediction(result) {
  const probability = Number(result.failure_probability || 0);
  const angle = Math.round(probability * 360);
  const riskColors = {
    low: '#57ffad',
    medium: '#ffd166',
    high: '#ff5c7a',
  };

  document.documentElement.style.setProperty('--risk-angle', `${angle}deg`);
  document.documentElement.style.setProperty('--green', riskColors[result.risk] || '#57ffad');
  document.getElementById('failureProbability').textContent = formatPercent(probability);
  document.getElementById('predictionLabel').textContent = titleCase(result.prediction);
  document.getElementById('confidenceLabel').textContent = `${result.confidence_percent}%`;
  document.getElementById('riskLabel').textContent = titleCase(result.risk);
  renderTrace(result.trace);
}

async function runPrediction(event) {
  event?.preventDefault();
  renderTrace([{ name: 'Running', value: 'Submitting build signals to Flask API' }]);
  const payload = collectPayload();
  const result = await fetchJson('/api/predict', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(payload),
  });
  renderPrediction(result);
}

async function init() {
  state.info = await fetchJson('/api/model-info');
  state.defaults = state.info.defaults;

  const formFields = document.getElementById('formFields');
  state.info.fields.forEach((field) => formFields.appendChild(createField(field)));
  setDefaults();

  renderMetrics(state.info);
  renderPipelineSteps(state.info);
  renderFeatureBars(state.info);
  renderArtifacts(state.info);

  document.getElementById('predictionForm').addEventListener('submit', runPrediction);
  document.getElementById('resetButton').addEventListener('click', () => {
    setDefaults();
    runPrediction();
  });

  await runPrediction();
}

init().catch((error) => {
  console.error(error);
  renderTrace([{ name: 'Frontend error', value: error.message }]);
});
