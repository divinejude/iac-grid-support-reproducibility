# IEEE 13-Node OpenDSS Benchmark

These files provide the IEEE 13-node distribution test feeder used by the main co-simulation demo.

- `IEEE13Nodeckt.dss`
- `IEEELineCodes.dss`
- `IEEE13Node_BusXY.csv`

The simulation adapter `IEEE13OpenDSSFleetFeeder` compiles `IEEE13Nodeckt.dss`, reads the original OpenDSS loads, and adds controllable TCL fleet loads at those same buses/phases. Aggregate TCL active and reactive commands are distributed proportional to original load kW.

Source references:

- IEEE PES Test Feeders: https://cmte.ieee.org/pes-testfeeders/
- OpenDSS IEEE 13-node example mirror: https://github.com/tshort/OpenDSS/tree/master/Distrib/IEEETestCases/13Bus
