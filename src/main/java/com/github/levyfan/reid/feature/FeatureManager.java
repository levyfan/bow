package com.github.levyfan.reid.feature;

import com.github.levyfan.reid.BowImage;

import java.io.IOException;
import java.net.URISyntaxException;

/**
 * @author fanliwen
 */
public class FeatureManager {

    private Feature[] features;

    public FeatureManager() throws IOException, URISyntaxException {
        this.features = new Feature[]{new Hsv(), new Cn(), new Hog(), new Siltp()};
    }

    public Feature[] getFeatures() {
        return this.features;
    }

    public void feature(BowImage bowImage) {
        for (Feature feature : features) {
            feature.extract(bowImage);
        }

        // release memory
        bowImage.image = null;
        bowImage.image4 = null;
        bowImage.mask = null;
        bowImage.mask4 = null;
    }
}
