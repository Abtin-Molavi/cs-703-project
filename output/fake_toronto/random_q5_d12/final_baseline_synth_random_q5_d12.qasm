OPENQASM 2.0;
include "qelib1.inc";
qreg q[27];
rz(2.412011) q[18];
rz(4.5153255) q[18];
rz(6.1815207) q[18];
rz(5.4547209) q[18];
rz(5.8728208) q[18];
rz(2.7303074) q[18];
rz(4.7214589) q[21];
cx q[21],q[18];
rz(0.49090649) q[23];
cx q[21],q[23];
cx q[23],q[21];
cx q[21],q[23];
cx q[24],q[23];
cx q[23],q[24];
cx q[24],q[23];
cx q[23],q[24];
rz(5.9220304) q[25];
rz(2.445447) q[25];
rz(5.0796944) q[25];
rz(0.4960646) q[25];
rz(0.32705788) q[25];
cx q[24],q[25];
rz(2.9122506) q[24];
cx q[24],q[23];
cx q[23],q[24];
cx q[24],q[23];
rz(5.8792143) q[25];
rz(4.6340834) q[25];
cx q[24],q[25];
cx q[23],q[24];
cx q[24],q[23];
cx q[23],q[24];
cx q[21],q[23];
cx q[18],q[21];
rz(1.5780224) q[18];
rz(2.9788202) q[21];
cx q[23],q[21];
cx q[24],q[23];
cx q[23],q[24];
cx q[24],q[23];
cx q[21],q[23];
rz(4.4190864) q[21];
cx q[23],q[24];
rz(3.6647608) q[24];
rz(0.76188099) q[24];
rz(6.2150178) q[24];
rz(1.4642231) q[24];
rz(3.2071874) q[24];
rz(5.6875611) q[24];
rz(0.80399538) q[24];
rz(3.1569366) q[25];
rz(4.3491615) q[25];
