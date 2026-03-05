/*
 * tb_top.v - Fornax M2 Testbench
 * Verifies the full Layer 0 Transformer Block.
 */

`timescale 1ns / 1ps

module tb_top;

    parameter DIM = __DIM__;
    parameter NUM_LAYERS = __LAYERS__;
    parameter IN_LEN = __IN_LEN__;
    parameter OUT_LEN = __OUT_LEN__;

    reg clk;
    reg rst_n;
    reg signed [7:0] in_data;
    reg in_valid;
    wire signed [31:0] out_data;
    wire out_valid;

    // Instantiate the Full Block
    fornax_layer0 dut (
        .clk(clk),
        .rst_n(rst_n),
        .in_data(in_data),
        .in_valid(in_valid),
        .out_data(out_data),
        .out_valid(out_valid)
    );

    reg [7:0] input_mem [0:IN_LEN-1];
    reg [31:0] expected_mem [0:OUT_LEN-1];
    integer i, outfile;
    integer captured;
    reg [63:0] wait_cycles;
    reg [63:0] timeout_cycles;

    always #5 clk = ~clk;

    initial begin
`ifdef FORNAX_DEBUG
        $dumpfile("top.vcd");
        $dumpvars(0, tb_top);
`endif
        
        $readmemh("../testvectors/input.hex", input_mem);
        $readmemh("../testvectors/expected.hex", expected_mem);
        outfile = $fopen("../testvectors/actual.hex", "w");

        clk = 0;
        rst_n = 0;
        in_data = 0;
        in_valid = 0;
        // Force 64-bit arithmetic to avoid overflow when OUT_LEN is large (e.g. lm_head vocab slices).
        timeout_cycles = (64'd1 * (DIM + OUT_LEN) * DIM * 20 * (NUM_LAYERS * 2));

        #100 rst_n = 1;
        #20;

        $display("[TB] Starting M2 Full Block Simulation...");
        
        // Feed input vector
        // In M2 top.v, modules are chained. We feed the vector once.
        for (i = 0; i < IN_LEN; i = i + 1) begin
            @(posedge clk);
            #1;
            in_data = input_mem[i];
            in_valid = 1;
        end
        @(posedge clk);
        #1;
        in_valid = 0;

        // Capture exactly OUT_LEN valid output samples.
        // matmul emits sparse valid pulses, so sample only when out_valid=1.
        captured = 0;
        wait_cycles = 0;
        while (captured < OUT_LEN) begin
            @(posedge clk);
            wait_cycles = wait_cycles + 1;

            if (out_valid) begin
                if (captured == 0)
                    $display("[TB] out_valid first detected at %t", $time);

                $fdisplay(outfile, "%08x", out_data);
                $fflush(outfile);

                if (captured == 0 || captured == OUT_LEN-1)
                    $display("[TB] Output %0d: %h (Expected %h)", captured, out_data, expected_mem[captured]);

                captured = captured + 1;
            end

            if (wait_cycles > timeout_cycles) begin
                $display("[TB] ERROR: Output capture timeout. captured=%0d/%0d", captured, OUT_LEN);
                $finish;
            end
        end

        #100;
        $fclose(outfile);
        $display("[TB] M2 Simulation Finished.");
        $finish;
    end

`ifdef FORNAX_DEBUG
    // Monitor out_valid status
    always #10000000 begin
        $display("[MON] t:%t | out_valid:%b | out_data:%h", $time, out_valid, out_data);
        $fflush();
    end
`endif

    // Global Timeout
    initial begin
        #(500000000000 * NUM_LAYERS);
        $display("[TB] FATAL: Global Timeout reached at %t!", $time);
        $finish;
    end

endmodule
