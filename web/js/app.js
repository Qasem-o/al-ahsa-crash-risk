const map = L.map('map', {
  zoomControl: true,
  preferCanvas: true,
}).setView([25.383, 49.586], 12.5);

L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
  attribution: '&copy; <a href="https://www.openstreetmap.org/copyright">OSM contributors</a>',
  maxZoom: 19,
}).addTo(map);

const API_BASE = (() => {
  if (window.location.port && window.location.port !== '8000') {
    return `${window.location.protocol}//${window.location.hostname}:8000`;
  }
  return '';
})();

const DEFAULT_WEIGHT = 4;
const HOVER_WEIGHT = 6;
const SELECTED_WEIGHT = 7;

const state = {
  layer: null,
  selectedLayer: null,
  boundsSet: false,
  weather: {
    condition: null,
    rain_mm: null,
    visibility_km: null,
    temperature_c: null,
  },
  debounceHandle: null,
};

const WEATHER_PRESETS = {
  clear: { rain_mm: 0, visibility_km: 15, temperature_c: 34 },
  rainy: { rain_mm: 6, visibility_km: 8, temperature_c: 29 },
  foggy: { rain_mm: 1.5, visibility_km: 5.5, temperature_c: 26 },
  dusty: { rain_mm: 0.4, visibility_km: 6.5, temperature_c: 37 },
  storm: { rain_mm: 10, visibility_km: 4, temperature_c: 25 },
};

const DEBOUNCE_DELAY = 250;

const weatherSelect = document.getElementById('weather-condition');
const rainToggle = document.getElementById('rain-toggle');
const rainRange = document.getElementById('rain-mm');
const rainValue = document.getElementById('rain-mm-value');
const visibilityToggle = document.getElementById('visibility-toggle');
const visibilityRange = document.getElementById('visibility-km');
const visibilityValue = document.getElementById('visibility-km-value');
const temperatureToggle = document.getElementById('temperature-toggle');
const temperatureRange = document.getElementById('temperature-c');
const temperatureValue = document.getElementById('temperature-c-value');
const resetButton = document.getElementById('reset-weather');

function buildQueryString() {
  const params = new URLSearchParams();
  if (state.weather.condition) params.set('weather_condition', state.weather.condition);
  if (state.weather.rain_mm !== null) params.set('rain_mm', state.weather.rain_mm.toString());
  if (state.weather.visibility_km !== null) params.set('visibility_km', state.weather.visibility_km.toString());
  if (state.weather.temperature_c !== null) params.set('temperature_c', state.weather.temperature_c.toString());
  return params.toString();
}

function buildApiUrl(path, query) {
  const qs = query ? `?${query}` : '';
  return `${API_BASE}${path}${qs}`;
}

function scheduleReload() {
  if (state.debounceHandle) clearTimeout(state.debounceHandle);
  state.debounceHandle = setTimeout(() => {
    loadSegments();
  }, DEBOUNCE_DELAY);
}

function normaliseText(value, fallback = '—') {
  if (Array.isArray(value)) {
    const joined = value.filter(Boolean).join(' / ');
    return joined || fallback;
  }
  if (value === null || value === undefined || value === '') return fallback;
  return String(value);
}

function escapeHtml(value) {
  if (value === null || value === undefined) {
    return '';
  }
  const div = document.createElement('div');
  div.textContent = String(value);
  return div.innerHTML;
}

function applyOverride(toggle, slider, label, key, value, unit, decimals = 1) {
  if (value === null || Number.isNaN(value)) {
    toggle.checked = false;
    slider.disabled = true;
    label.textContent = 'Baseline';
    state.weather[key] = null;
    return;
  }

  toggle.checked = true;
  slider.disabled = false;
  slider.value = value;
  state.weather[key] = Number(value);
  label.textContent = `${Number(value).toFixed(decimals)} ${unit}`;
}

function updateNumericOverride(slider, label, key, unit, decimals = 1) {
  const value = Number(slider.value);
  state.weather[key] = value;
  label.textContent = `${value.toFixed(decimals)} ${unit}`;
  scheduleReload();
}

function setWeatherCondition(condition) {
  state.weather.condition = condition || null;

  if (!condition) {
    applyOverride(rainToggle, rainRange, rainValue, 'rain_mm', null, 'mm');
    applyOverride(visibilityToggle, visibilityRange, visibilityValue, 'visibility_km', null, 'km');
    applyOverride(temperatureToggle, temperatureRange, temperatureValue, 'temperature_c', null, '°C');
  } else if (WEATHER_PRESETS[condition]) {
    const preset = WEATHER_PRESETS[condition];
    applyOverride(rainToggle, rainRange, rainValue, 'rain_mm', preset.rain_mm, 'mm');
    applyOverride(
      visibilityToggle,
      visibilityRange,
      visibilityValue,
      'visibility_km',
      preset.visibility_km,
      'km'
    );
    applyOverride(
      temperatureToggle,
      temperatureRange,
      temperatureValue,
      'temperature_c',
      preset.temperature_c,
      '°C'
    );
  }

  scheduleReload();
}

function resetWeatherOverrides(triggerReload = true) {
  weatherSelect.value = '';
  state.weather.condition = null;
  applyOverride(rainToggle, rainRange, rainValue, 'rain_mm', null, 'mm');
  applyOverride(visibilityToggle, visibilityRange, visibilityValue, 'visibility_km', null, 'km');
  applyOverride(temperatureToggle, temperatureRange, temperatureValue, 'temperature_c', null, '°C');

  if (triggerReload) scheduleReload();
}

async function loadSegments() {
  const query = buildQueryString();
  const url = buildApiUrl('/api/segments.geojson', query);
  try {
    const response = await fetch(url);
    if (!response.ok) {
      throw new Error(`Failed to load segments: ${response.status}`);
    }
    const data = await response.json();
    renderSegments(data);
  } catch (error) {
    console.error(error);
    alert('Unable to load crash predictions. Ensure the backend is running.');
  }
}

function getRiskColor(percent) {
  if (percent >= 8) return '#b30000';
  if (percent >= 5) return '#e34a33';
  if (percent >= 3) return '#fc8d59';
  if (percent >= 1.5) return '#fdcc8a';
  return '#fef0d9';
}

function buildTooltipContent(feature) {
  const props = feature.properties || {};
  const name = escapeHtml(normaliseText(props.name, 'Unnamed road'));
  const highway = escapeHtml(normaliseText(props.highway, '—'));
  const percent = props.predicted_crash_percent === null || props.predicted_crash_percent === undefined
    ? '—'
    : `${Number(props.predicted_crash_percent).toFixed(2)}%`;

  return `<div class="tooltip-content"><strong>${name}</strong><br />${highway}<br />Risk: ${percent}</div>`;
}

function highlightLayer(layer) {
  if (state.selectedLayer && state.selectedLayer !== layer) {
    state.selectedLayer.setStyle({ weight: DEFAULT_WEIGHT });
  }
  state.selectedLayer = layer;
  layer.setStyle({ weight: SELECTED_WEIGHT });
}

function resetSelectedLayer() {
  if (state.selectedLayer) {
    state.selectedLayer.setStyle({ weight: DEFAULT_WEIGHT });
    state.selectedLayer = null;
  }
}

function renderSegments(geojson) {
  if (state.layer) {
    state.layer.remove();
  }
  resetSelectedLayer();

  state.layer = L.geoJSON(geojson, {
    style: feature => ({
      color: getRiskColor(feature.properties.predicted_crash_percent ?? 0),
      weight: DEFAULT_WEIGHT,
      opacity: 0.85,
    }),
    onEachFeature: (feature, layer) => {
      const tooltipHtml = buildTooltipContent(feature);
      layer.bindTooltip(tooltipHtml, { sticky: true, direction: 'top', opacity: 0.92 });

      layer.on('click', () => handleSegmentClick(feature.properties.segment_id, layer, feature.properties));
      layer.on('mouseover', () => {
        if (state.selectedLayer !== layer) {
          layer.setStyle({ weight: HOVER_WEIGHT });
        }
      });
      layer.on('mouseout', () => {
        if (state.selectedLayer !== layer) {
          layer.setStyle({ weight: DEFAULT_WEIGHT });
        }
      });
    },
  }).addTo(map);

  if (!state.boundsSet) {
    const bounds = state.layer.getBounds();
    if (bounds.isValid()) {
      map.fitBounds(bounds, { padding: [30, 30] });
      map.setMaxBounds(bounds.pad(0.25));
      state.boundsSet = true;
    }
  }
}

async function handleSegmentClick(segmentId, layer, properties = {}) {
  if (layer) {
    highlightLayer(layer);
  }

  updateInfoPanel({
    segment_id: segmentId,
    name: properties.name,
    highway: properties.highway,
    length_m: properties.length || properties.length_m,
    predicted_crash_percent: properties.predicted_crash_percent,
  });

  const query = buildQueryString();
  const url = buildApiUrl(`/api/segments/${segmentId}`, query);

  try {
    const response = await fetch(url);
    if (!response.ok) {
      throw new Error('Segment details unavailable');
    }
    const data = await response.json();
    updateInfoPanel(data);
  } catch (error) {
    console.error(error);
    alert('Unable to retrieve segment details.');
  }
}

function updateInfoPanel(info) {
  document.getElementById('segment-id').textContent = info.segment_id;
  document.getElementById('segment-name').textContent = normaliseText(info.name);
  document.getElementById('segment-highway').textContent = normaliseText(info.highway);
  const lengthValue = Number(info.length_m ?? info.length);
  document.getElementById('segment-length').textContent = Number.isFinite(lengthValue)
    ? lengthValue.toFixed(1)
    : '—';
  const percentValue = Number(info.predicted_crash_percent);
  document.getElementById('segment-risk').textContent = Number.isFinite(percentValue)
    ? `${percentValue.toFixed(2)}%`
    : '—';
}

function initialiseControls() {
  weatherSelect.addEventListener('change', event => {
    setWeatherCondition(event.target.value);
  });

  rainToggle.addEventListener('change', () => {
    if (rainToggle.checked) {
      rainRange.disabled = false;
      updateNumericOverride(rainRange, rainValue, 'rain_mm', 'mm');
    } else {
      rainRange.disabled = true;
      rainValue.textContent = 'Baseline';
      state.weather.rain_mm = null;
      scheduleReload();
    }
  });
  rainRange.addEventListener('input', () => {
    if (!rainRange.disabled) {
      updateNumericOverride(rainRange, rainValue, 'rain_mm', 'mm');
    }
  });

  visibilityToggle.addEventListener('change', () => {
    if (visibilityToggle.checked) {
      visibilityRange.disabled = false;
      updateNumericOverride(visibilityRange, visibilityValue, 'visibility_km', 'km');
    } else {
      visibilityRange.disabled = true;
      visibilityValue.textContent = 'Baseline';
      state.weather.visibility_km = null;
      scheduleReload();
    }
  });
  visibilityRange.addEventListener('input', () => {
    if (!visibilityRange.disabled) {
      updateNumericOverride(visibilityRange, visibilityValue, 'visibility_km', 'km');
    }
  });

  temperatureToggle.addEventListener('change', () => {
    if (temperatureToggle.checked) {
      temperatureRange.disabled = false;
      updateNumericOverride(temperatureRange, temperatureValue, 'temperature_c', '°C');
    } else {
      temperatureRange.disabled = true;
      temperatureValue.textContent = 'Baseline';
      state.weather.temperature_c = null;
      scheduleReload();
    }
  });
  temperatureRange.addEventListener('input', () => {
    if (!temperatureRange.disabled) {
      updateNumericOverride(temperatureRange, temperatureValue, 'temperature_c', '°C');
    }
  });

  resetButton.addEventListener('click', () => {
    resetWeatherOverrides();
  });

  resetWeatherOverrides(false);
}

initialiseControls();
createLegend();
loadSegments();

function createLegend() {
  const legend = L.control({ position: 'bottomright' });
  legend.onAdd = () => {
    const div = L.DomUtil.create('div', 'legend');
    const grades = [0, 1.5, 3, 5, 8];
    const labels = [
      '< 1.5%',
      '1.5% - 3%',
      '3% - 5%',
      '5% - 8%',
      '≥ 8%',
    ];

    div.innerHTML = '<h3>Crash Risk</h3>';
    grades.forEach((grade, index) => {
      const upper = grades[index + 1] ?? grade + 5;
      const sample = (grade + upper) / 2;
      const color = getRiskColor(sample);
      const span = document.createElement('span');
      span.innerHTML = `<i style="background:${color}"></i>${labels[index]}`;
      div.appendChild(span);
    });
    return div;
  };
  legend.addTo(map);
}
