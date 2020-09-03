import React , {Component} from 'react';
import {Link} from "react-router-dom";

class Header extends Component{
    render() {
        return(
            <nav className="navbar navbar-expand-lg navbar-dark bg-dark">
                <a className="navbar-brand" href="#">Just ML things</a>
                <button className="navbar-toggler" type="button" data-toggle="collapse" data-target="#navbarText"
                        aria-controls="navbarText" aria-expanded="false" aria-label="Toggle navigation">
                    <span className="navbar-toggler-icon"></span>
                </button>
                <div className="collapse navbar-collapse" id="navbarText">
                    <ul className="navbar-nav mr-auto">
                        <li className="nav-item active">
                            <Link to={'/'} className="nav-link">
                            Home <span className="sr-only">(current)</span>
                                </Link>
                        </li>
                        <li className="nav-item">
                            <Link to="/assignment2MNIST" className="nav-link" >
                                Mnist
                            </Link>
                        </li>
                        <li className="nav-item">
                            <Link to='/assignment2cifar' className="nav-link">
                                Cifar10
                                </Link>
                        </li>
                    </ul>


                </div>
            </nav>
        )
    }
}
export default Header;
