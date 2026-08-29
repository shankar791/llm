# ISRO Evaluation Data Access Guide
**Project:** SatQuery AI - SIH26167
**Last Updated:** August 29, 2026

## 📡 Data Sources

ISRO evaluation data is **restricted** and requires formal access through Bhuvan/NRSC portal.

### Cartosat-2S (Optical)
| Product | Resolution | Access Portal | Status |
|---------|-----------|---------------|--------|
| Cartosat-2S PAN | 0.65m | Bhuvan/NRSC | ⬜ Pending MoU |
| Cartosat-2S MX (4-band) | 2m | Bhuvan/NRSC | ⬜ Pending MoU |
| Cartosat-2S MX (merged) | 0.65m | Bhuvan/NRSC | ⬜ Pending MoU |

### RISAT (SAR)
| Product | Band | Resolution | Access Portal | Status |
|---------|------|-----------|---------------|--------|
| RISAT-1 MRS | C-band | 3-50m | Bhuvan/NRSC | ⬜ Pending MoU |
| RISAT-1 CRS | C-band | 25m | Bhuvan/NRSC | ⬜ Pending MoU |
| RISAT-1 FRS-1 | C-band | 3m | Bhuvan/NRSC | ⬜ Pending MoU |
| RISAT-1 FRS-2 | C-band | 9m | Bhuvan/NRSC | ⬜ Pending MoU |
| RISAT-2B | X-band | 1-3m | Bhuvan/NRSC | ⬜ Pending MoU |

## 🎯 Recommended AOIs (5-10 test sites)

For comprehensive evaluation, we need data covering diverse scenarios:

### Urban Areas
1. **Delhi NCR** - Dense urban, rapid development
2. **Mumbai** - Coastal urban, mixed land use
3. **Bengaluru** - Tech corridor, urban sprawl

### Agriculture
4. **Punjab** - Intensive agriculture, crop monitoring
5. **Maharashtra (Vidarbha)** - Drought-prone agriculture

### Forest & Environment
6. **Western Ghats** - Dense forest, deforestation
7. **Northeast India** - Forest cover change

### Water Bodies
8. **Chilika Lake** - Coastal lagoon, water quality
9. **Ganga River (Varanasi)** - River dynamics

### Disaster/Change
10. **Chamoli (Uttarakhand)** - Landslide/flash flood affected

## 📋 Access Process

### Step 1: Institutional Registration
- [ ] Register on **Bhuvan portal**: https://bhuvan.nrsc.gov.in/
- [ ] Faculty initiates **ISRO/SAC collaboration request**
- [ ] Sign **MoU with NRSC** (National Remote Sensing Centre)
- [ ] Get **SAC (Space Applications Centre) approval** for specific products

### Step 2: Data Request Submission
- [ ] Define **specific AOI coordinates** (shapefiles)
- [ ] Specify **date ranges** and **acquisition parameters**
- [ ] Submit via **Bhuvan Open Data Archive** or **ISRO Data Centre**
- [ ] Wait for **approval (typically 2-4 weeks)**

### Step 3: Download & Process
- [ ] Download via **secure FTP** or **API**
- [ ] Verify data integrity (checksums)
- [ ] Organize by sensor/date/AOI
- [ ] Create metadata catalog

## 🛠️ Folder Structure

```
data/isro_evaluation/
├── cartosat_2s_pan/          # 0.65m panchromatic
│   ├── metadata/
│   ├── raw/
│   └── processed/
├── cartosat_2s_ms/           # 2m multispectral
│   ├── metadata/
│   ├── raw/
│   └── processed/
├── risat_1/                  # C-band SAR
│   ├── metadata/
│   ├── raw/
│   └── processed/
├── risat_2b/                 # X-band SAR
│   ├── metadata/
│   ├── raw/
│   └── processed/
├── aoi_definitions/          # Shapefiles for test sites
│   ├── urban/
│   ├── agriculture/
│   ├── forest/
│   ├── water/
│   └── disaster/
└── evaluation_pairs/         # Optical-SAR pairs for fusion testing
    ├── cartosat_risat1/
    ├── cartosat_risat2b/
    └── bitemporal/
```

## 🚀 Quick Start (Pending Data)

While waiting for ISRO access, use the **Mendeley merged samples** already available:
- Location: `data/mendeley_merged_sar_optical/`
- Contains: Optical + HH SAR + HV SAR + DEM
- Use for: Pipeline testing, agentic controller validation

## 📞 Key Contacts

- **ISRO Data Centre**: https://www.isro.gov.in/
- **NRSC**: https://www.nrsc.gov.in/
- **Bhuvan Portal**: https://bhuvan.nrsc.gov.in/
- **SAC (Ahmedabad)**: https://www.sac.gov.in/

## ✅ Action Items (Priority Order)

1. **URGENT**: Faculty to initiate MoU process (deadline: 2 weeks)
2. Define 5-10 AOI shapefiles (can use Bhuvan WMS for visualization)
3. Test pipeline with Mendeley data
4. Document access procedure for future teams
5. Set up secure storage (data is large: ~1-10TB)

## 📊 Expected Data Volume

| Product | Size per AOI | 10 AOIs Total |
|---------|--------------|---------------|
| Cartosat-2S PAN | ~5-10 GB | ~50-100 GB |
| Cartosat-2S MS | ~2-5 GB | ~20-50 GB |
| RISAT-1 | ~1-3 GB | ~10-30 GB |
| RISAT-2B | ~2-5 GB | ~20-50 GB |
| **Total** | | **~100-230 GB** |

Plan for **~250GB storage** for full evaluation dataset.