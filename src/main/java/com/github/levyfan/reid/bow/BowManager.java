package com.github.levyfan.reid.bow;

import com.github.levyfan.reid.App;
import com.github.levyfan.reid.BowImage;
import com.github.levyfan.reid.feature.Feature;
import com.github.levyfan.reid.feature.FeatureManager;

/**
 * @author fanliwen
 */
public class BowManager {

    private Bow bow;
    private FeatureManager featureManager;

    public BowManager(Bow bow, FeatureManager featureManager) {
        this.bow = bow;
        this.featureManager = featureManager;
    }

    public void bow(BowImage bowImage) {
        featureManager.feature(bowImage);
        
        for (Feature.Type type : App.types) {
            bow.bow(bowImage, type);
        }
        bow.bow(bowImage, Feature.Type.ALL);
    }
}
