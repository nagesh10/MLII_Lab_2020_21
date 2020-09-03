import React , {Component} from "react";
import {render} from "react-dom";
import {HashRouter as Router , Route,Switch} from "react-router-dom";

import Home from "./Home";
import Header from "./layout/header";
import Assignment2Cifar from "./layout/Cifar10";
import "./App.css";
import Assignment2MNIST from "./layout/mnist";

class App extends Component{
    constructor(props) {
        super(props);
        this.state = {
            data:[],
            loaded:false,
            placeholder:'Loading'
        };
    }

      render() {
    return (

     <Router>
         <Header/>
         <Switch>
             <Route  exact path={'/'} component={Home} />
             <Route  exact path ={'/assignment2Cifar'}  component={Assignment2Cifar} />
             <Route  exact path ={'/assignment2MNIST'}  component={Assignment2MNIST} />
         </Switch>
     </Router>

    );
  }

}

export default App;
const container=document.getElementById("app");
render(<App/>,container);
