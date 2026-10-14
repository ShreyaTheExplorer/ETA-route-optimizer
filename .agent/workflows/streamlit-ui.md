---
description: Person D - Streamlit app with map, ETA list and before/after
---
1. Read AGENTS.md and docs/CONTRACT.md. The UI calls the API only through requests.post(f"{API_URL}/optimize").
2. Create frontend/app.py. Sidebar: depot lat/lon, departure date and time, algorithm select, stop input by (a) table editor with lat/lon rows, (b) "generate N random stops in NYC" button with a seed, (c) click-on-map to add stops if streamlit-folium is available.
3. Main area: Optimize button, then two tabs or two columns, Before (baseline_route) and After (route), each a map with numbered markers and route lines. Use folium + streamlit-folium or pydeck.
4. Show metrics: total_minutes before and after, improvement_pct, and a per-stop table (order, id, leg_minutes, eta_minutes, arrival_time) exactly as returned.
5. Handle errors: show the API's 422 message in a friendly box; show a spinner while waiting.
6. If the API is not ready, add a --mock mode that returns a fake response in the CONTRACT shape so UI work is not blocked.
7. Run it and show me a screenshot description of each state.
