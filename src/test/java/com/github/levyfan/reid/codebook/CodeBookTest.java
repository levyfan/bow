package com.github.levyfan.reid.codebook;

import com.jmatio.io.MatFileReader;
import com.jmatio.types.MLDouble;
import org.junit.Test;

import java.io.File;
import java.io.IOException;
import java.net.URISyntaxException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

/**
 * @author fanliwen
 */
public class CodeBookTest {

    @Test
    public void test() throws URISyntaxException, IOException {
        File file = new File(this.getClass().getResource("/kmeans.mat").toURI());
        MatFileReader reader = new MatFileReader(file);
        MLDouble array = (MLDouble) reader.getMLArray("X");

        List<double[]> feature = new ArrayList<>(array.getM());
        for (int m = 0; m < array.getM(); m++) {
            double[] v = new double[array.getN()];
            for (int n = 0; n < array.getN(); n++) {
                v[n] = array.getReal(m, n);
            }
            feature.add(v);
        }

        CodeBook codeBook = new CodeBook(3);

        List<double[]> words_fast = codeBook.codebook(feature);
        for (double[] word : words_fast) {
            System.out.println(Arrays.toString(word));
        }

        //  5.62608695652174	2.04782608695652
        //  4.29259259259259	1.35925925925926
        //  1.46200000000000	0.246000000000000
    }
}
