# IEEE 118 Public Dynamic Benchmark Data

This folder contains the public IEEE 118 RAW/DYR pair used by the Mini
Grid-Mind IEEE118 M1+M2 benchmark path.

Files:

- `ieee118.raw`
- `ieee118.dyr`

Source:

- UC3M e-Archivo dataset mirror:
  `https://e-archivo.uc3m.es/entities/publication/a0d15c41-c528-49cf-adc4-d44b1950ceea`
- Dataset name: `Transient Stability Constrained Optimal Power Flow GAMS model generator on Python`

Integrity checks:

```text
47210f890b938e0115c4f1e8ca57031c  ieee118.raw
98edb8e83aab29106837deadee89de8d  ieee118.dyr
```

These files are public benchmark data, not customer-validated dynamic data.
The current M2 interconnection model still represents new projects as static
PQ injections/loads and does not add detailed inverter, protection, controller,
or synchronous-machine dynamics for the new project.
